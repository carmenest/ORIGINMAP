import typer
from originmap.pipeline.bootstrap import bootstrap
from originmap.pipeline.download import download_meteorites
from originmap.pipeline.ingest import ingest_meteorites
from originmap.pipeline.metrics import compute_metrics
from originmap.pipeline.visualize import visualize as run_visualize
from originmap.pipeline.report import build_report
from originmap.utils.logging import setup_logger
from originmap.utils.provenance import create_manifest
from originmap.config import PROJECT_ROOT, REPORTS

app = typer.Typer(help="ORIGINMAP — computational lab for panspermia research")

@app.command()
def init():
    """
    Inicializa el laboratorio ORIGINMAP (estructura + manifest).
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("🧬 Initializing ORIGINMAP laboratory...")
    manifest = bootstrap()
    typer.echo(f"✔ Laboratory ready")
    typer.echo(f"✔ Manifest created at: {manifest}")
    typer.echo(f"✔ Log file: {log_file}")


@app.command()
def download():
    """
    Descarga datasets públicos (meteoritos).
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("⬇️ Downloading meteorite dataset...")
    info = download_meteorites()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"meteorites": info},
        parameters={"step": "download"},
    )

    typer.echo(f"✔ Downloaded: {info['file']}")
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def ingest():
    """
    Normaliza datasets crudos a formato procesado.
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("🧪 Ingesting meteorite dataset...")
    info = ingest_meteorites()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"processed_dataset": info},
        parameters={"step": "ingest"},
    )

    typer.echo(f"✔ Processed rows: {info['rows']}")
    typer.echo(f"✔ Output: {info['output_file']}")
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def metrics():
    """
    Calcula métricas objetivas del dataset procesado.
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("📊 Computing objective metrics...")
    info = compute_metrics()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"metrics_outputs": info},
        parameters={"step": "metrics"},
    )

    typer.echo(f"✔ Metrics summary: {info['summary']}")
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def visualize():
    """
    Genera visualizaciones científicas reproducibles.
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("📈 Generating scientific visualizations...")
    outputs = run_visualize()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"visualizations": outputs},
        parameters={"step": "visualize"},
    )

    for o in outputs:
        typer.echo(f"✔ Created: {o}")

    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def report():
    """
    Genera informe HTML reproducible (sin narrativa).
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("🧾 Building reproducible report...")
    out = build_report()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"report": out},
        parameters={"step": "report"},
    )

    typer.echo(f"✔ Report created: {out}")
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def observe(
    full_pipeline: bool = typer.Option(False, "--full", "-f", help="Run full pipeline before observing"),
):
    """
    Ejecuta ciclo de observación: detecta anomalías y registra hallazgos.
    """
    from originmap.analysis.observer import run_observation_cycle

    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("🔭 Running observation cycle...")

    result = run_observation_cycle(run_pipeline=full_pipeline)

    typer.echo(f"✔ Observation ID: {result['observation_id']}")
    typer.echo(f"✔ Anomalies detected: {result['anomaly_count']}")
    typer.echo(f"✔ JSON: {result['json_path']}")
    typer.echo(f"✔ Markdown: {result['md_path']}")

    comp = result["comparison"]
    if not comp["is_first_run"]:
        typer.echo(f"📊 Comparison with previous runs:")
        typer.echo(f"   Stable anomalies: {len(comp['stable_anomalies'])}")
        typer.echo(f"   New anomalies: {len(comp['new_anomalies'])}")
        typer.echo(f"   Resolved: {len(comp['resolved_anomalies'])}")

        if comp["new_anomalies"]:
            typer.echo("⚠️  New anomalies to investigate:")
            for a in comp["new_anomalies"]:
                typer.echo(f"   - {a}")

    typer.echo(f"✔ Log: {log_file}")


@app.command()
def battery(
    null_spec: str = typer.Option("1-5", "--null", "-N", help="Null models to run (e.g., '1-5', '2,4', '1-3,5')"),
    bins: str = typer.Option("10", "--bins", "-b", help="Mass bin configs (e.g., '8,10,12,20')"),
    stats: str = typer.Option("cv,varlog,mad_ratio", "--stats", "-S", help="Statistics to compute"),
    n_perms: int = typer.Option(500, "--n", "-n", help="Number of permutations"),
    n_bootstrap: int = typer.Option(200, "--bootstrap", "-B", help="Number of bootstrap samples"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    min_samples: int = typer.Option(30, "--min", "-m", help="Minimum samples per class"),
    fdr: float = typer.Option(0.10, "--fdr", help="FDR threshold for significance"),
    subsample: int = typer.Option(100, "--subsample", help="Subsample size for Null-5"),
):
    """
    Ejecuta batería completa de modelos nulos.

    Null models:
      1: Global permutation (baseline)
      2: Mass-bin stratified
      3: Mass × Time stratified
      4: Mass × Fall/Found stratified
      5: Balanced subsampling

    Statistics:
      cv: Coefficient of Variation
      varlog: Variance of log(mass)
      mad_ratio: MAD / Median

    Examples:
      originmap battery --null 1-5 --bins 8,10,12,20 --n 500
      originmap battery --null 2-4 --stats cv,varlog
    """
    from originmap.analysis.hypothesis_battery import run_battery, save_battery_report

    log_file = setup_logger(PROJECT_ROOT / "logs")

    # Parse inputs
    bin_configs = [int(b.strip()) for b in bins.split(",")]
    stat_names = [s.strip() for s in stats.split(",")]

    typer.echo("=" * 60)
    typer.echo("ORIGINMAP — Null Model Battery")
    typer.echo("=" * 60)
    typer.echo(f"Null models: {null_spec}")
    typer.echo(f"Statistics: {stat_names}")
    typer.echo(f"Bin configs: {bin_configs}")
    typer.echo(f"Permutations: {n_perms}, Bootstrap: {n_bootstrap}")
    typer.echo()

    report = run_battery(
        null_spec=null_spec,
        stat_names=stat_names,
        bin_configs=bin_configs,
        n_permutations=n_perms,
        n_bootstrap=n_bootstrap,
        seed=seed,
        min_samples=min_samples,
        fdr_threshold=fdr,
        subsample_size=subsample,
    )

    files = save_battery_report(report)

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"battery_report": files},
        parameters={
            "null_spec": null_spec,
            "bins": bin_configs,
            "stats": stat_names,
            "n_permutations": n_perms,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
    )

    typer.echo()
    typer.echo(f"✔ Robust candidates: {len(report.robust_candidates)}")
    if report.robust_candidates:
        for rc in report.robust_candidates:
            typer.echo(f"   ★ {rc}")
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def hypothesis(
    experiment: str = typer.Argument("O-D2", help="Experiment ID (O-D2 = mass heterogeneity)"),
    n_perms: int = typer.Option(500, "--n", "-n", help="Number of permutations"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
):
    """
    Ejecuta test de hipótesis con modelo nulo por permutación.

    O-D2: Mass heterogeneity test - are mass distributions within classes
    different from random assignment?
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo(f"🔬 Running hypothesis test: {experiment}")
    typer.echo(f"   Permutations: {n_perms}, Seed: {seed}")

    if experiment == "battery":
        from originmap.analysis.hypothesis_battery import run_quick_battery
        result = run_quick_battery(n_permutations=n_perms, seed=seed)

        typer.echo(f"✔ Results: {result['csv']}")
        typer.echo(f"✔ Summary: {result['json']}")
        typer.echo(f"✔ Plot: {result['plot']}")
        typer.echo(f"✔ Report: {result['report']}")
        typer.echo(f"✔ Robust candidates (survive ALL nulls): {result['robust_count']}")

        if result["robust_candidates"]:
            typer.echo("★ ROBUST CANDIDATES:")
            for rc in result["robust_candidates"]:
                typer.echo(f"   {rc}")

    elif experiment == "O-D3":
        from originmap.analysis.hypothesis_stratified import run_experiment_o_delta_3
        result = run_experiment_o_delta_3(n_permutations=n_perms, seed=seed)

        typer.echo(f"✔ Results: {result['csv']}")
        typer.echo(f"✔ Summary: {result['summary']}")
        typer.echo(f"✔ Plot: {result['plot']}")
        typer.echo(f"✔ Discoveries (FDR≤0.05): {result['discoveries']['fdr_005']}")
        typer.echo(f"✔ Discoveries (FDR≤0.10): {result['discoveries']['fdr_010']}")

        if result["survivors"]:
            typer.echo("🌟 SURVIVED from O-Δ2 (genuine structure):")
            for item in result["survivors"]:
                typer.echo(f"   {item['recclass']}: z={item['cv_zscore']:.2f}, q={item['cv_fdr_q']:.4f}")

        if result["fallen"]:
            typer.echo("💀 FALLEN from O-Δ2 (was artifact):")
            for item in result["fallen"]:
                typer.echo(f"   {item['recclass']}: z={item['cv_zscore']:.2f}, q={item['cv_fdr_q']:.4f}")

    elif experiment == "O-D2":
        from originmap.analysis.hypothesis_mass import run_experiment_o_delta_2
        result = run_experiment_o_delta_2(n_permutations=n_perms, seed=seed)

        typer.echo(f"✔ Results: {result['csv']}")
        typer.echo(f"✔ Summary: {result['summary']}")
        typer.echo(f"✔ Plot: {result['plot']}")
        typer.echo(f"✔ Discoveries (FDR≤0.05): {result['discoveries']['cv_fdr_005']}")
        typer.echo(f"✔ Discoveries (FDR≤0.10): {result['discoveries']['cv_fdr_010']}")

        if result["tight_classes"]:
            typer.echo("🔵 Unusually TIGHT mass distribution (low variance):")
            for item in result["tight_classes"]:
                typer.echo(f"   {item['recclass']}: z={item['cv_zscore']:.2f}, q={item['cv_fdr_q']:.4f}")

        if result["dispersed_classes"]:
            typer.echo("🔴 Unusually DISPERSED mass distribution (high variance):")
            for item in result["dispersed_classes"]:
                typer.echo(f"   {item['recclass']}: z={item['cv_zscore']:.2f}, q={item['cv_fdr_q']:.4f}")
    else:
        from originmap.analysis.hypothesis import run_experiment_o_delta_1
        result = run_experiment_o_delta_1(n_permutations=n_perms, seed=seed)
        typer.echo(f"✔ Results: {result['csv']}")

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"hypothesis_test": result},
        parameters={"experiment": experiment, "n_permutations": n_perms, "seed": seed},
    )

    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def sample_threshold(
    sizes: str = typer.Option("30,50,75,100,150,200,300,500", "--sizes", "-s", help="Sample sizes to test"),
    classes: str = typer.Option("L6,H6,H4,Ureilite,H5,L5,LL6,LL5", "--classes", "-c", help="Target classes"),
    stats: str = typer.Option("cv,varlog,mad", "--stats", "-S", help="Statistics to test"),
    n_perms: int = typer.Option(200, "--perm", "-p", help="Permutations per test"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    alpha: float = typer.Option(0.05, "--alpha", "-a", help="Significance threshold"),
):
    """
    O-Δ8: Find sample size threshold for structure emergence.

    Tests at what N the 'structure' in meteorite classes becomes significant.
    Answers: Is structure real or a sample size artifact?
    """
    from originmap.analysis.hypothesis_sample_threshold import run_full_o_delta_8

    log_file = setup_logger(PROJECT_ROOT / "logs")

    # Parse inputs
    sample_sizes = [int(s.strip()) for s in sizes.split(",")]
    target_classes = [c.strip() for c in classes.split(",")]
    stat_names = [s.strip() for s in stats.split(",")]

    typer.echo("=" * 60)
    typer.echo("O-Δ8: Sample Size Threshold Analysis")
    typer.echo("=" * 60)
    typer.echo(f"Sample sizes: {sample_sizes}")
    typer.echo(f"Classes: {target_classes}")
    typer.echo(f"Statistics: {stat_names}")
    typer.echo()

    files = run_full_o_delta_8(
        sample_sizes=sample_sizes,
        target_classes=target_classes,
        stat_names=stat_names,
        n_permutations=n_perms,
        seed=seed,
        alpha=alpha,
    )

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"sample_threshold": files},
        parameters={
            "experiment": "O-D8",
            "sample_sizes": sample_sizes,
            "target_classes": target_classes,
            "stat_names": stat_names,
            "n_permutations": n_perms,
            "seed": seed,
            "alpha": alpha,
        },
    )

    typer.echo()
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def stability_clusters(
    n_clusters: int = typer.Option(4, "--clusters", "-k", help="Number of clusters"),
    method: str = typer.Option("ward", "--method", "-m", help="Linkage method (ward, complete, average)"),
    mode: str = typer.Option("binary", "--mode", help="Matrix mode (binary or zscore)"),
):
    """
    O-Δ7: Cluster meteorite classes by stability profiles.

    Identifies distinct stability regimes based on survival patterns
    across the null model battery tests.

    Requires: Run 'originmap battery' first.
    """
    from originmap.analysis.hypothesis_stability_clustering import run_o_delta_7

    log_file = setup_logger(PROJECT_ROOT / "logs")

    typer.echo("=" * 60)
    typer.echo("O-Δ7: Stability Regime Clustering")
    typer.echo("=" * 60)
    typer.echo(f"Clusters: {n_clusters}, Method: {method}, Mode: {mode}")
    typer.echo()

    files = run_o_delta_7(
        n_clusters=n_clusters,
        method=method,
        mode=mode,
    )

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"stability_clustering": files},
        parameters={
            "experiment": "O-D7",
            "n_clusters": n_clusters,
            "method": method,
            "mode": mode,
        },
    )

    typer.echo()
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def grade_curve(
    n_perms: int = typer.Option(1000, "--perm", "-p", help="Number of permutations"),
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b", help="Number of bootstrap samples"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
):
    """
    O-Δ6: Test heterogeneity curvature vs petrologic grade.

    Tests H-FRAG-2 hypothesis: Grade 5 is an interior maximum.

    H0: H(G) is monotonic or flat
    H1: H(5) > H(4) AND H(5) > H(6)

    Metrics:
      varlog: var(log(mass))
      mad: MAD(log(mass)) / median(log(mass))
    """
    from originmap.analysis.hypothesis_grade_curve import run_o_delta_6

    log_file = setup_logger(PROJECT_ROOT / "logs")

    typer.echo("=" * 60)
    typer.echo("O-Δ6: Heterogeneity Curvature vs Petrologic Grade")
    typer.echo("=" * 60)
    typer.echo(f"Permutations: {n_perms}, Bootstrap: {n_bootstrap}, Seed: {seed}")
    typer.echo()

    files = run_o_delta_6(
        n_permutations=n_perms,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"grade_curve": files},
        parameters={
            "experiment": "O-D6",
            "n_permutations": n_perms,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
    )

    typer.echo()
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def falls():
    """
    O-Δ10: Falls-Only Analysis — The Unbiased Sample.

    Falls (meteorites we saw fall) have no collection bias.
    This is our best window into the TRUE flux of meteorites.

    Hypotheses tested:
      H-FALL-1: Genuine mass structure in Falls
      H-FALL-2: Class proportions differ Falls vs Finds
      H-FALL-3: Seasonality in Falls (orbital signature)
    """
    from originmap.analysis.hypothesis_falls import run_full_o_delta_10

    log_file = setup_logger(PROJECT_ROOT / "logs")

    typer.echo("=" * 60)
    typer.echo("O-Δ10: Falls-Only Analysis — The Unbiased Sample")
    typer.echo("=" * 60)
    typer.echo()

    files = run_full_o_delta_10()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"falls_analysis": files},
        parameters={
            "experiment": "O-D10",
        },
    )

    typer.echo()
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def temporal():
    """
    O-Δ9: Catalog Archaeology — Temporal Dynamics.

    Analyzes how the meteorite catalog evolved over time.

    Hypotheses tested:
      H-TEMP-1: Classes appear in "waves" (not uniformly)
      H-TEMP-2: Mean discovery mass decreases over time
      H-TEMP-3: Antarctica distorts the catalog post-1970

    Eras analyzed:
      - Classical (pre-1900)
      - Modern pre-Antarctica (1900-1969)
      - Antarctica boom (1970-1999)
      - Satellite era (2000+)
    """
    from originmap.analysis.hypothesis_temporal import run_full_o_delta_9

    log_file = setup_logger(PROJECT_ROOT / "logs")

    typer.echo("=" * 60)
    typer.echo("O-Δ9: Catalog Archaeology — Temporal Dynamics")
    typer.echo("=" * 60)
    typer.echo()

    files = run_full_o_delta_9()

    manifest = create_manifest(
        output_dir=REPORTS,
        inputs={"temporal_analysis": files},
        parameters={
            "experiment": "O-D9",
            "eras": ["classical", "modern_pre_antarctica", "antarctica_boom", "satellite_era"],
        },
    )

    typer.echo()
    typer.echo(f"✔ Manifest: {manifest}")
    typer.echo(f"✔ Log: {log_file}")


@app.command()
def full_cycle():
    """
    Ejecuta pipeline completo + observación (para cron automático).
    """
    log_file = setup_logger(PROJECT_ROOT / "logs")
    typer.echo("🔄 Starting full automated cycle...")

    # Pipeline
    typer.echo("⬇️ Downloading...")
    download_meteorites()

    typer.echo("🧪 Ingesting...")
    ingest_meteorites()

    typer.echo("📊 Computing metrics...")
    compute_metrics()

    typer.echo("📈 Visualizing...")
    run_visualize()

    typer.echo("🧾 Building report...")
    build_report()

    # Observation
    from originmap.analysis.observer import run_observation_cycle
    typer.echo("🔭 Observing...")
    result = run_observation_cycle(run_pipeline=False)

    typer.echo(f"✔ Full cycle complete")
    typer.echo(f"✔ Observation: {result['observation_id']}")
    typer.echo(f"✔ Anomalies: {result['anomaly_count']}")
    typer.echo(f"✔ Log: {log_file}")


def main():
    app()

if __name__ == "__main__":
    main()
