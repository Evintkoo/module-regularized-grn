/// Chapter figure generation for Phase 8 dissertation chapter.
/// Generates four SVG figures into figures/.
use anyhow::Result;
use std::fs;

fn main() -> Result<()> {
    fs::create_dir_all("figures")?;
    generate_architecture_diagram()?;
    generate_auroc_comparison()?;
    generate_pruning_curve()?;
    generate_compression_scatter()?;
    println!("✓ All chapter figures written to figures/");
    Ok(())
}

// ── Fig 4.1: Architecture Diagram ────────────────────────────────────────────

fn generate_architecture_diagram() -> Result<()> {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="780" height="420" font-family="Arial,sans-serif" font-size="13">
  <!-- Background -->
  <rect width="780" height="420" fill="white" stroke="none"/>

  <!-- Title -->
  <text x="390" y="30" text-anchor="middle" font-size="15" font-weight="bold">Two-Tower vs Cross-Encoder Architecture</text>

  <!-- ── TWO-TOWER (left panel) ────────────────────────── -->
  <text x="195" y="60" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a6bb5">Two-Tower MLP</text>

  <!-- TF Input -->
  <rect x="60" y="80" width="120" height="36" rx="6" fill="#dbeeff" stroke="#1a6bb5" stroke-width="1.5"/>
  <text x="120" y="103" text-anchor="middle" fill="#1a6bb5">TF: embed(512) ‖ expr(11)</text>

  <!-- Gene Input -->
  <rect x="210" y="80" width="130" height="36" rx="6" fill="#dbeeff" stroke="#1a6bb5" stroke-width="1.5"/>
  <text x="275" y="103" text-anchor="middle" fill="#1a6bb5">Gene: embed(512) ‖ expr(11)</text>

  <!-- TF FC1 -->
  <rect x="75" y="152" width="90" height="30" rx="4" fill="#b3d4f5" stroke="#1a6bb5" stroke-width="1.5"/>
  <text x="120" y="172" text-anchor="middle">FC(512) + ReLU</text>

  <!-- Gene FC1 -->
  <rect x="225" y="152" width="100" height="30" rx="4" fill="#b3d4f5" stroke="#1a6bb5" stroke-width="1.5"/>
  <text x="275" y="172" text-anchor="middle">FC(512) + ReLU</text>

  <!-- TF FC2 -->
  <rect x="75" y="218" width="90" height="30" rx="4" fill="#b3d4f5" stroke="#1a6bb5" stroke-width="1.5"/>
  <text x="120" y="238" text-anchor="middle">FC(512)</text>

  <!-- Gene FC2 -->
  <rect x="225" y="218" width="100" height="30" rx="4" fill="#b3d4f5" stroke="#1a6bb5" stroke-width="1.5"/>
  <text x="275" y="238" text-anchor="middle">FC(512)</text>

  <!-- z_TF label -->
  <text x="120" y="280" text-anchor="middle" font-style="italic" fill="#1a6bb5">z_TF ∈ ℝ⁵¹²</text>

  <!-- z_Gene label -->
  <text x="275" y="280" text-anchor="middle" font-style="italic" fill="#1a6bb5">z_Gene ∈ ℝ⁵¹²</text>

  <!-- Cosine similarity node -->
  <ellipse cx="197" cy="320" rx="60" ry="22" fill="#fff3b0" stroke="#c8a000" stroke-width="2"/>
  <text x="197" y="325" text-anchor="middle" font-weight="bold">cos(z_TF, z_Gene)/τ</text>

  <!-- Score output -->
  <rect x="152" y="368" width="90" height="28" rx="4" fill="#d4edda" stroke="#28a745" stroke-width="1.5"/>
  <text x="197" y="387" text-anchor="middle">σ(score) ∈ [0,1]</text>

  <!-- Arrows: TF path -->
  <line x1="120" y1="116" x2="120" y2="152" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="120" y1="182" x2="120" y2="218" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="120" y1="248" x2="160" y2="302" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Arrows: Gene path -->
  <line x1="275" y1="116" x2="275" y2="152" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="275" y1="182" x2="275" y2="218" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="275" y1="248" x2="234" y2="302" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Arrow: cosine to score -->
  <line x1="197" y1="342" x2="197" y2="368" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Param count -->
  <text x="197" y="412" text-anchor="middle" font-size="11" fill="#555">5,581,824 parameters</text>

  <!-- ── DIVIDER ───────────────────────────────────────── -->
  <line x1="390" y1="50" x2="390" y2="410" stroke="#ccc" stroke-width="1.5" stroke-dasharray="6,4"/>

  <!-- ── CROSS-ENCODER (right panel) ──────────────────── -->
  <text x="585" y="60" text-anchor="middle" font-size="14" font-weight="bold" fill="#b5341a">Cross-Encoder MLP</text>

  <!-- Joint input box -->
  <rect x="450" y="80" width="270" height="52" rx="6" fill="#ffe5de" stroke="#b5341a" stroke-width="1.5"/>
  <text x="585" y="101" text-anchor="middle" fill="#b5341a">[TF_emb ‖ Gene_emb ‖</text>
  <text x="585" y="120" text-anchor="middle" fill="#b5341a">TF_emb⊙Gene_emb ‖ expr] = 1558-dim</text>

  <!-- FC1 -->
  <rect x="510" y="170" width="150" height="30" rx="4" fill="#ffc4b5" stroke="#b5341a" stroke-width="1.5"/>
  <text x="585" y="190" text-anchor="middle">FC(512) + ReLU</text>

  <!-- FC2 -->
  <rect x="510" y="236" width="150" height="30" rx="4" fill="#ffc4b5" stroke="#b5341a" stroke-width="1.5"/>
  <text x="585" y="256" text-anchor="middle">FC(512) + ReLU</text>

  <!-- FC3 -->
  <rect x="535" y="302" width="100" height="30" rx="4" fill="#ffc4b5" stroke="#b5341a" stroke-width="1.5"/>
  <text x="585" y="322" text-anchor="middle">FC(1, logit)</text>

  <!-- Score output CE -->
  <rect x="540" y="368" width="90" height="28" rx="4" fill="#d4edda" stroke="#28a745" stroke-width="1.5"/>
  <text x="585" y="387" text-anchor="middle">σ(logit) ∈ [0,1]</text>

  <!-- Arrows CE -->
  <line x1="585" y1="132" x2="585" y2="170" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="585" y1="200" x2="585" y2="236" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="585" y1="266" x2="585" y2="302" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="585" y1="332" x2="585" y2="368" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Param count CE -->
  <text x="585" y="412" text-anchor="middle" font-size="11" fill="#555">5,581,313 parameters</text>

  <!-- Arrow marker definition -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#555"/>
    </marker>
  </defs>
</svg>"##;

    fs::write("figures/fig4_1_architecture.svg", svg)?;
    println!("  ✓ fig4_1_architecture.svg");
    Ok(())
}

// ── Fig 4.2: AUROC Comparison Bar Chart ──────────────────────────────────────

fn generate_auroc_comparison() -> Result<()> {
    use plotters::prelude::*;

    let path = "figures/fig4_2_auroc_comparison.svg";
    let root = SVGBackend::new(path, (600, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    // Data: (label, auroc, color)
    let bars: &[(&str, f64, RGBColor)] = &[
        ("Two-Tower\n1:1", 0.8097, RGBColor(70, 130, 200)),
        ("Two-Tower\n5:1", 0.7434, RGBColor(150, 190, 230)),
        ("Cross-Encoder\n1:1", 0.9040, RGBColor(210, 80, 60)),
        ("Cross-Encoder\n5:1", 0.9150, RGBColor(240, 150, 130)),
    ];

    let mut chart = ChartBuilder::on(&root)
        .caption("AUROC by Model and Negative Sampling Ratio", ("sans-serif", 16).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(
            ["Two-Tower\n1:1", "Two-Tower\n5:1", "Cross-Encoder\n1:1", "Cross-Encoder\n5:1"]
                .into_segmented(),
            0.65f64..1.0f64,
        )?;

    chart.configure_mesh()
        .y_desc("AUROC")
        .y_label_formatter(&|v| format!("{:.2}", v))
        .draw()?;

    for (label, auroc, color) in bars {
        chart.draw_series(
            std::iter::once(Rectangle::new(
                [(SegmentValue::Exact(label), 0.65), (SegmentValue::CenterOf(label), *auroc)],
                color.filled(),
            )),
        )?;
        // Value label on top of bar
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.4}", auroc),
            (SegmentValue::CenterOf(label), auroc + 0.005),
            ("sans-serif", 11).into_font(),
        )))?;
    }

    root.present()?;
    println!("  ✓ fig4_2_auroc_comparison.svg");
    Ok(())
}

// ── Fig 4.3: Sparsity vs AUROC Retention ─────────────────────────────────────

fn generate_pruning_curve() -> Result<()> {
    use plotters::prelude::*;

    // Data from results/neuron_pruning_results.json
    let sparsity:   &[f64] = &[0.0,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];
    let posthoc:    &[f64] = &[1.0000, 1.0000, 0.9996, 0.9998, 0.9998, 1.0005, 1.0016, 1.0014, 1.0033, 1.0043, 1.0079, 1.0104, 1.0027];
    let finetuned:  &[f64] = &[1.0229, 1.0013, 1.0038, 0.9866, 1.0044, 0.9901, 1.0144, 1.0064, 0.9625, 0.9815, 1.0059, 0.9659, 1.0248];

    let path = "figures/fig4_3_pruning_curve.svg";
    let root = SVGBackend::new(path, (640, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("AUROC Retention vs Sparsity (Two-Tower)", ("sans-serif", 16).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f64..0.92f64, 0.93f64..1.05f64)?;

    chart.configure_mesh()
        .x_desc("Sparsity (fraction of neurons removed per tower)")
        .y_desc("AUROC Retention (pruned / baseline)")
        .x_label_formatter(&|v| format!("{:.0}%", v * 100.0))
        .y_label_formatter(&|v| format!("{:.3}", v))
        .draw()?;

    // Baseline reference at 1.0
    chart.draw_series(LineSeries::new(
        vec![(0.0, 1.0), (0.90, 1.0)],
        BLACK.mix(0.4).stroke_width(1),
    ))?.label("Baseline (1.000)").legend(|(x, y)| {
        PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.4).stroke_width(1))
    });

    // 95% retention threshold
    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.95), (0.90, 0.95)],
        RED.mix(0.3).stroke_width(1),
    ))?.label("95% retention threshold").legend(|(x, y)| {
        PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.4).stroke_width(1))
    });

    // Post-hoc line
    let ph_points: Vec<(f64, f64)> = sparsity.iter().zip(posthoc.iter()).map(|(&x, &y)| (x, y)).collect();
    chart.draw_series(LineSeries::new(ph_points.clone(), RGBColor(70, 130, 200).stroke_width(2)))?
        .label("Post-hoc (no fine-tuning)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(70, 130, 200).stroke_width(2)));
    chart.draw_series(ph_points.iter().map(|&(x, y)| {
        Circle::new((x, y), 4, RGBColor(70, 130, 200).filled())
    }))?;

    // Fine-tuned line
    let ft_points: Vec<(f64, f64)> = sparsity.iter().zip(finetuned.iter()).map(|(&x, &y)| (x, y)).collect();
    chart.draw_series(LineSeries::new(ft_points.clone(), RGBColor(210, 80, 60).stroke_width(2)))?
        .label("Fine-tuned (10 epochs)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(210, 80, 60).stroke_width(2)));
    chart.draw_series(ft_points.iter().map(|&(x, y)| {
        Circle::new((x, y), 4, RGBColor(210, 80, 60).filled())
    }))?;

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK.mix(0.3))
        .draw()?;

    root.present()?;
    println!("  ✓ fig4_3_pruning_curve.svg");
    Ok(())
}

// ── Fig 4.4: Compression Ratio vs Post-hoc AUROC ─────────────────────────────

fn generate_compression_scatter() -> Result<()> {
    use plotters::prelude::*;

    // (compression_ratio, posthoc_auroc, sparsity_label)
    let points: &[(f64, f64, &str)] = &[
        (1.0000, 0.8015, "0%"),
        (0.9903, 0.8015, "5%"),
        (0.9811, 0.8012, "10%"),
        (0.9714, 0.8014, "15%"),
        (0.9621, 0.8014, "20%"),
        (0.9525, 0.8019, "25%"),
        (0.9428, 0.8028, "30%"),
        (0.9239, 0.8027, "40%"),
        (0.9050, 0.8042, "50%"),
        (0.8860, 0.8050, "60%"),
        (0.8671, 0.8078, "70%"),
        (0.8478, 0.8099, "80%"),
        (0.8289, 0.8037, "90%"),
    ];

    let path = "figures/fig4_4_compression_scatter.svg";
    let root = SVGBackend::new(path, (600, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Compression vs Post-hoc AUROC", ("sans-serif", 16).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(65)
        .build_cartesian_2d(0.81f64..1.02f64, 0.790f64..0.820f64)?;

    chart.configure_mesh()
        .x_desc("Compression Ratio (params_remaining / baseline_params)")
        .y_desc("Post-hoc AUROC")
        .x_label_formatter(&|v| format!("{:.2}", v))
        .y_label_formatter(&|v| format!("{:.4}", v))
        .draw()?;

    // Baseline AUROC reference line
    chart.draw_series(LineSeries::new(
        vec![(0.81, 0.8015), (1.02, 0.8015)],
        BLACK.mix(0.35).stroke_width(1),
    ))?.label("Baseline AUROC (0.8015)").legend(|(x, y)| {
        PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.4).stroke_width(1))
    });

    // Scatter points colored by sparsity (darker = more sparse)
    for (i, &(cr, auroc, label)) in points.iter().enumerate() {
        let intensity = 200u8.saturating_sub((i as u8) * 14);
        let color = RGBColor(30, intensity / 2, intensity);
        chart.draw_series(std::iter::once(Circle::new((cr, auroc), 6, color.filled())))?;
        // Label every other point to avoid clutter
        if i % 2 == 0 || i == points.len() - 1 {
            chart.draw_series(std::iter::once(Text::new(
                label.to_string(),
                (cr + 0.002, auroc + 0.0002),
                ("sans-serif", 10).into_font(),
            )))?;
        }
    }

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK.mix(0.3))
        .draw()?;

    root.present()?;
    println!("  ✓ fig4_4_compression_scatter.svg");
    Ok(())
}
