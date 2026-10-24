// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) lapoule.dev

//! Does creating passes leak? The process footprint over many submissions of
//! render passes, of compute passes, and of the block encoder's real work.
//!
//! tuile's viewer builds imagery mips on the CPU because Apple's Metal driver was
//! seen to leak in proportion to the number of **render passes** created
//! (gfx-rs/wgpu#8768). The block encoder adds one **compute pass** per upload
//! batch, so the same question is asked of compute passes here, with the same
//! instrument for both.
//!
//! ```text
//! cargo run --release --features gpu --example pass-footprint -- render  20000 4
//! cargo run --release --features gpu --example pass-footprint -- compute 20000 4
//! cargo run --release --features gpu --example pass-footprint -- encode  20000 1
//! ```
//!
//! Arguments: the kind of pass, the number of submissions, the passes per
//! submission. It prints the footprint (`footprint -p`, macOS; resident size
//! from `ps` elsewhere) every tenth of the run, and the growth per thousand
//! passes once the first tenth has warmed every pool up.

use std::process::Command;

use texcomp::gpu::{feature, texture_format, GpuEncoder, Level, GPU_FORMATS};

fn samples(adapter: &wgpu::Adapter, format: texcomp::BlockFormat) -> bool {
    adapter.features().contains(feature(format))
        && adapter
            .get_texture_format_features(texture_format(format, true))
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

fn footprint_kib() -> Option<f64> {
    let pid = std::process::id().to_string();
    if cfg!(target_os = "macos") {
        let out = Command::new("footprint")
            .args(["-p", &pid, "-f", "bytes", "--noCategories"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().find(|l| l.contains("Footprint:"))?;
        let rest = line.split("Footprint:").nth(1)?.trim();
        let mut parts = rest.split_whitespace();
        let value: f64 = parts.next()?.parse().ok()?;
        let unit = parts.next()?;
        Some(match unit {
            "B" => value / 1024.0,
            "KB" => value,
            "MB" => value * 1024.0,
            "GB" => value * 1024.0 * 1024.0,
            _ => return None,
        })
    } else {
        let out = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kind = args.first().map(String::as_str).unwrap_or("compute");
    let submissions: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let per: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let (adapter, device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("an adapter");
        let features = GPU_FORMATS
            .into_iter()
            .filter(|f| samples(&adapter, *f))
            .fold(wgpu::Features::empty(), |acc, f| acc | feature(f));
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("a device");
        (adapter, device, queue)
    });
    let info = adapter.get_info();
    println!(
        "{:?} {} — {kind}, {submissions} submissions × {per} passes",
        info.backend, info.name
    );

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n\
             @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i: vec3u) { b[i.x] = i.x; }"
                .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let encoder_format = GPU_FORMATS.into_iter().find(|f| samples(&adapter, *f));
    let encoder = encoder_format.map(|f| GpuEncoder::new(&device, f));
    let tile: Vec<u8> = (0..256 * 256 * 4).map(|i| (i * 7 % 251) as u8).collect();
    if kind == "encode" && encoder.is_none() {
        eprintln!("no block format the encoder writes on this adapter");
        std::process::exit(2);
    }

    let tenth = (submissions / 10).max(1);
    let mut warm: Option<f64> = None;
    for s in 0..submissions {
        let mut cmd = device.create_command_encoder(&Default::default());
        for _ in 0..per {
            match kind {
                "render" => {
                    let _pass = cmd.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                "compute" => {
                    let mut pass = cmd.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                "encode" => {
                    let level = [Level {
                        width: 256,
                        height: 256,
                        rgba8: &tile,
                    }];
                    let enc = encoder.as_ref().expect("checked above");
                    let _ = enc.encode(&device, &queue, &mut cmd, &[&level[..]], true);
                }
                other => panic!("unknown kind {other}: render, compute or encode"),
            }
        }
        queue.submit([cmd.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        if (s + 1) % tenth == 0 {
            let kib = footprint_kib().unwrap_or(f64::NAN);
            let passes = (s + 1) as f64 * per as f64;
            match warm {
                None => {
                    warm = Some(kib);
                    println!("{:>8} passes: {:>10.0} KiB (warm)", passes, kib);
                }
                Some(base) => {
                    let since = passes - tenth as f64 * per as f64;
                    println!(
                        "{:>8} passes: {:>10.0} KiB  {:+8.1} KiB per 1000 passes since warm",
                        passes,
                        kib,
                        (kib - base) / since * 1000.0
                    );
                }
            }
        }
    }
}
