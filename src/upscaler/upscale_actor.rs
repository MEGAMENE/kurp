use std::sync::Arc;

use bytes::Bytes;
use image::ImageFormat;
use log::{error, info};
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort, SupervisionEvent};

use crate::config::app_config::AppConfig;
use crate::upscaler::upscaler::{RealCuganUpscaler, Upscaler};

pub enum UpscaleSupervisorMessage {
    Upscale(Bytes, ImageFormat, RpcReplyPort<(Bytes, ImageFormat)>),
    Init(Arc<AppConfig>),
    Destroy,
}

pub enum UpscaleMessage {
    Upscale(Bytes, ImageFormat, RpcReplyPort<(Bytes, ImageFormat)>),
}

pub struct UpscaleSupervisorActor;

pub struct SupervisorState {
    config: Option<Arc<AppConfig>>,
    upscale_actor: Option<ActorRef<UpscaleMessage>>,
}

#[async_trait::async_trait]
impl Actor for UpscaleSupervisorActor {
    type Msg = UpscaleSupervisorMessage;
    type State = SupervisorState;
    type Arguments = ();

    async fn pre_start(&self, _myself: ActorRef<Self::Msg>, _args: Self::Arguments) -> Result<Self::State, ActorProcessingErr> {
        Ok(SupervisorState { config: None, upscale_actor: None })
    }

    async fn handle(&self, myself: ActorRef<Self::Msg>, message: Self::Msg, state: &mut Self::State) -> Result<(), ActorProcessingErr> {
        match message {
            UpscaleSupervisorMessage::Upscale(image, format, reply_to) => {
                match &state.upscale_actor {
                    None => { return Err(From::from("Upscale Actor is not Initialized")); }
                    Some(upscale_actor) => {
                        let _ = upscale_actor
                            .send_message(UpscaleMessage::Upscale(image, format, reply_to));
                    }
                }
            }

            UpscaleSupervisorMessage::Init(config) => {
                if let Some(actor) = &state.upscale_actor {
                    actor.stop(None);
                }

                let (upscale_actor, _) = Actor::spawn_linked(
                    None,
                    UpscaleActor,
                    config.clone(),
                    myself.get_cell(),
                ).await?;
                state.config = Some(config);
                state.upscale_actor = Some(upscale_actor);
            }

            UpscaleSupervisorMessage::Destroy => {
                if state.config.is_none() && state.upscale_actor.is_none() { return Ok(()); }
                state.upscale_actor.as_ref().unwrap().stop(None);
                state.config = None;
                state.upscale_actor = None;
            }
        }

        Ok(())
    }

    async fn handle_supervisor_evt(&self, myself: ActorRef<Self::Msg>, message: SupervisionEvent, state: &mut Self::State) -> Result<(), ActorProcessingErr> {
        match message {
            SupervisionEvent::ActorFailed(cell, panic_msg) => {
                error!("Upscale actor {cell:?} panicked with '{panic_msg}'");
                info!("Restarting Upscale actor");
                match &state.config {
                    None => { error!("Config is not Initialized") }
                    Some(config) => {
                        let (upscale_actor, _) = Actor::spawn_linked(
                            None,
                            UpscaleActor,
                            config.clone(),
                            myself.get_cell(),
                        ).await?;
                        state.upscale_actor = Some(upscale_actor);
                    }
                };
            }
            _ => {}
        }

        Ok(())
    }
}

pub struct UpscaleActor;

#[async_trait::async_trait]
impl Actor for UpscaleActor {
    type Msg = UpscaleMessage;
    type State = Box<dyn Upscaler>;
    type Arguments = Arc<AppConfig>;

    async fn pre_start(&self, _myself: ActorRef<Self::Msg>, args: Self::Arguments) -> Result<Self::State, ActorProcessingErr> {
        let upscaler: Box<dyn Upscaler> = Box::new(RealCuganUpscaler::new(args));

        Ok(upscaler)
    }

    async fn handle(&self, _myself: ActorRef<Self::Msg>, message: Self::Msg, state: &mut Self::State) -> Result<(), ActorProcessingErr> {
        match message {
            UpscaleMessage::Upscale(image, format, reply_to) => {
                let _ = reply_to.send(state.upscale(image, format));
            }
        }

        Ok(())
    }
}