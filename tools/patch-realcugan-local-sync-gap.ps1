$ErrorActionPreference = 'Stop'
$source = 'realcugan-ncnn-vulkan-rs/src/realcugan-ncnn-vulkan/src/realcugan.cpp'
$text = Get-Content $source -Raw
$start = $text.IndexOf('int RealCUGAN::process_se_sync_gap(')
$end = $text.IndexOf("`nint RealCUGAN::process_se_very_rough_stage0(", $start)
if ($start -lt 0 -or $end -lt 0) { throw 'Could not locate process_se_sync_gap in Real-CUGAN source.' }

$replacement = @'
#pragma message("KURP EXPERIMENT: local 3x3 sync-gap patch compiled")
int RealCUGAN::process_se_sync_gap(const ncnn::Mat& inimage, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const
{
    const int w = inimage.w;
    const int h = inimage.h;
    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
    const int tta_count = tta_mode ? 8 : 1;
    const int tiles = ytiles * xtiles * tta_count;

    std::vector< std::vector<ncnn::VkMat> > feats(names.size());
    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                if (tta_mode)
                {
                    for (int ti = 0; ti < 8; ti++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);
                        feats[i].push_back(feat);
                    }
                }
                else
                {
                    ncnn::VkMat feat;
                    cache.load(yi, xi, 0, names[i], feat);
                    feats[i].push_back(feat);
                }
            }
        }
    }

    ncnn::VkCompute cmd(vkdev);
    std::vector< std::vector<ncnn::Mat> > feats_cpu(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        feats_cpu[i].resize(tiles);
        for (int j = 0; j < tiles; j++)
            cmd.record_download(feats[i][j], feats_cpu[i][j], opt);
    }

    cmd.submit_and_wait();
    cmd.reset();

    for (size_t i = 0; i < names.size(); i++)
    {
        for (int j = 0; j < tiles; j++)
        {
            if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && feats_cpu[i][j].elembits() == 16)
            {
                ncnn::Mat feat_fp32;
                ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                feats_cpu[i][j] = feat_fp32;
            }
            if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1)
            {
                ncnn::Mat feat_cpu_unpacked;
                ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                feats_cpu[i][j] = feat_cpu_unpacked;
            }
        }
    }

    // Preserve local structure: each tile receives the average of itself and
    // its immediate 3x3 tile neighborhood instead of one global average.
    std::vector< std::vector<ncnn::VkMat> > synced_feats(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        synced_feats[i].resize(tiles);
        for (int yi = 0; yi < ytiles; yi++)
        {
            for (int xi = 0; xi < xtiles; xi++)
            {
                for (int ti = 0; ti < tta_count; ti++)
                {
                    const int center = (yi * xtiles + xi) * tta_count + ti;
                    ncnn::Mat avgfeat;
                    avgfeat.create_like(feats_cpu[i][center]);
                    avgfeat.fill(0.f);

                    int count = 0;
                    for (int ny = std::max(0, yi - 1); ny <= std::min(ytiles - 1, yi + 1); ny++)
                    {
                        for (int nx = std::max(0, xi - 1); nx <= std::min(xtiles - 1, xi + 1); nx++)
                        {
                            const int index = (ny * xtiles + nx) * tta_count + ti;
                            const ncnn::Mat& f = feats_cpu[i][index];
                            const int len = avgfeat.total();
                            for (int k = 0; k < len; k++)
                                avgfeat[k] += f[k];
                            count++;
                        }
                    }

                    const int len = avgfeat.total();
                    for (int k = 0; k < len; k++)
                        avgfeat[k] /= count;

                    cmd.record_upload(avgfeat, synced_feats[i][center], opt);
                }
            }
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                if (tta_mode)
                {
                    for (int ti = 0; ti < 8; ti++)
                        cache.save(yi, xi, ti, names[i], synced_feats[i][(yi * xtiles + xi) * 8 + ti]);
                }
                else
                {
                    cache.save(yi, xi, 0, names[i], synced_feats[i][yi * xtiles + xi]);
                }
            }
        }
    }

    return 0;
}
'@

$text = $text.Substring(0, $start) + $replacement + $text.Substring($end)
Set-Content -Path $source -Value $text -NoNewline

# Force Cargo/CMake to rebuild the native Real-CUGAN code from the patched source.
Remove-Item target -Recurse -Force -ErrorAction SilentlyContinue

Write-Host 'Applied local 3x3 sync-gap averaging patch.'
Write-Host 'Cleared target/ to force a native rebuild.'
Write-Host 'The C++ compiler should emit: KURP EXPERIMENT: local 3x3 sync-gap patch compiled'
