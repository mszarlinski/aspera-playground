#!/usr/bin/env python3
"""
Aspera benchmark & recommender (sequential with ETA + per-test logging)

- Runs a matrix of settings (ascp/ascp4, target rate, policy, sessions, split-threshold).
- Uses the SAME dataset for every test via a generated --file-list from --src.
- Each test runs SEQUENTIALLY (no cross-test parallelism). Multi-session tests
  launch N ascp processes in parallel to emulate real multi-session behavior.
- Writes:
  * console progress with ETA (updated each test)
  * per-test log files (stdout/err + metadata)
  * CSV:  aspera_benchmark_results_<timestamp>.csv
  * MD:   aspera_benchmark_report_<timestamp>.md
  * master run log with summary

Requirements:
  - Python 3.8+
  - ascp in PATH (and ascp4 if you pass --ascp4)
  - SSH key or other auth usable by Aspera CLI
"""

import argparse, os, sys, subprocess, shlex, time, tempfile, random, csv, json, re, atexit, shutil, signal
from datetime import datetime
from pathlib import Path
from shutil import which
from statistics import median, mean, pstdev

# ---------- helpers ----------
def hum_bytes(n):
    units = ["B","KB","MB","GB","TB","PB","EB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def hum_rate(bps):
    return f"{hum_bytes(bps)}/s"

def hum_time(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def list_files(root):
    files=[]
    for p, _, fnames in os.walk(root):
        for f in fnames:
            fp=os.path.join(p,f)
            try:
                sz=os.path.getsize(fp)
            except Exception:
                continue
            files.append((fp,sz))
    return files

def pick_sample(files, max_files=None, max_bytes=None):
    if (max_files is None and max_bytes is None) or (max_files==0 and (max_bytes is None or max_bytes==0)):
        return files[:]  # full dataset
    files_sorted = sorted(files, key=lambda x:x[1], reverse=True)
    if not files_sorted: return []
    mfiles = max_files if max_files and max_files>0 else len(files_sorted)
    mbytes = max_bytes if max_bytes and max_bytes>0 else (1<<62)
    split = max(1, int(len(files_sorted)*0.4))
    bigs   = files_sorted[:split]
    smalls = files_sorted[split:]
    cand = []
    cand.extend(bigs[:max(1, mfiles//2)])
    if smalls:
        cand.extend(random.sample(smalls, k=min(len(smalls), max(1, mfiles - len(cand)))))
    if len(cand) < mfiles:
        cand.extend(files_sorted[len(cand):mfiles])
    out=[]; total=0
    for fp,sz in cand:
        if len(out) >= mfiles: break
        out.append((fp,sz)); total+=sz
        if total>=mbytes: break
    if not out:
        out = files_sorted[:min(mfiles, len(files_sorted))]
    return out

def write_filelist(filelist, path):
    with open(path, 'w', encoding='utf-8') as f:
        for fp,_ in filelist:
            # Posix-style paths are safer for CLI consumption
            f.write(Path(fp).as_posix() + '\n')

def slugify(s):
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s).strip('-')
    return s[:120]

def build_ascp_cmd(tool, dest_user, dest_host, dest_path, ssh_key, rate_mbps, policy, filelist_path,
                   session_id=None, session_count=None, udp_port=None, split_threshold_mb=None, extra_flags=None):
    """
    Returns list[str] command for a single ascp/ascp4 process.
    For classic ascp multi-session, caller will generate N such commands with -C nid:ncount and unique -O.
    """
    cmd = [tool, "-QT", "-r", "-d"]  # quiet text, recursive, ensure dest directory exists
    if ssh_key:
        cmd += ["-i", ssh_key]
    if rate_mbps:
        cmd += ["-l", f"{int(rate_mbps)}m"]
    if policy:
        cmd += [f"--policy={policy}"]
    if tool == "ascp" and session_id and session_count:
        cmd += ["-C", f"{session_id}:{session_count}"]
    if tool == "ascp" and udp_port:
        cmd += ["-O", str(udp_port)]
    if tool == "ascp" and split_threshold_mb is not None:
        if int(split_threshold_mb) == 0:
            cmd += ["--multi-session-threshold=0"]
        else:
            cmd += ["--multi-session-threshold", str(int(split_threshold_mb)*1024*1024)]
    if extra_flags:
        cmd += extra_flags

    # file list
    if tool == "ascp4":
        cmd += [f"--file-list={filelist_path}", "--mode=send", "--user", dest_user, "--host", dest_host, dest_path.rstrip('/') + "/"]
    else:
        cmd += [f"--file-list={filelist_path}", f"{dest_user}@{dest_host}:{dest_path.rstrip('/')}/"]
    return cmd

def parse_bytes_transferred(log_text: str) -> int:
    """
    Attempts to parse total bytes transferred from ascp/ascp4 logs.
    Adjust/extend patterns as needed for your environment.
    """
    pats = [
        r"bytes\s+transferred:\s*([\d,]+)",
        r"Completed:\s*transferred\s*([\d,]+)\s*bytes",
        r"Transferred\s*([\d,]+)\s*bytes",
        r"total\s*bytes:\s*([\d,]+)",
    ]
    for p in pats:
        m = re.search(p, log_text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except Exception:
                continue
    return 0

def redact_cmd_for_log(cmd_list):
    # Redact obvious secrets if ever present in extra flags in the future
    redacted = []
    skip_next = False
    secret_flags = {"--pass", "--token"}
    for tok in cmd_list:
        if skip_next:
            redacted.append("REDACTED")
            skip_next = False
            continue
        if tok in secret_flags:
            redacted.append(tok); skip_next = True; continue
        redacted.append(tok)
    return redacted

def run_parallel(cmds, log_path, timeout=None):
    """
    Launch commands in parallel, capture combined outputs per-process in sequence,
    enforce optional timeout, and cleanly terminate on timeout/interrupt.
    """
    procs=[]
    start=time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"[meta] start_utc={datetime.utcnow().isoformat()}Z\n")
        for c in cmds:
            lf.write("[cmd] " + " ".join(shlex.quote(x) for x in redact_cmd_for_log(c)) + "\n")
        lf.write("\n[output]\n")
        lf.flush()

        # launch
        for c in cmds:
            procs.append(subprocess.Popen(
                c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            ))

        def kill_all(sig=signal.SIGTERM):
            for p in procs:
                if p and p.poll() is None:
                    try:
                        p.send_signal(sig)
                    except Exception:
                        pass

        outputs=[]
        rc=0
        try:
            for p in procs:
                try:
                    chunk = p.communicate(timeout=timeout if (timeout and timeout>0) else None)[0]
                except subprocess.TimeoutExpired:
                    # Timeout: try graceful then forceful termination of all
                    kill_all(signal.SIGTERM)
                    time.sleep(2)
                    kill_all(signal.SIGKILL)
                    chunk = ""
                    rc = rc or 124  # timeout code
                outputs.append(chunk)
                lf.write(chunk)
                lf.flush()
                rc = rc or (p.returncode if p.returncode is not None else 0)
        except KeyboardInterrupt:
            kill_all(signal.SIGTERM)
            time.sleep(1)
            kill_all(signal.SIGKILL)
            raise
        finally:
            # Best-effort cleanup
            kill_all(signal.SIGTERM)

        elapsed=time.perf_counter()-start
        lf.write(f"\n[meta] rc={rc} elapsed_seconds={elapsed:.3f}\n")
    return rc, "".join(outputs), elapsed

def make_table(rows, headers):
    colw = [len(h) for h in headers]
    for r in rows:
        for i,val in enumerate(r):
            colw[i] = max(colw[i], len(str(val)))
    line = "+".join("-"*(w+2) for w in colw)
    out=[]
    out.append(line)
    out.append("|" + "|".join(" " + headers[i].ljust(colw[i]) + " " for i in range(len(headers))) + "|")
    out.append(line)
    for r in rows:
        out.append("|" + "|".join(" " + str(r[i]).ljust(colw[i]) + " " for i in range(len(headers))) + "|")
    out.append(line)
    return "\n".join(out)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Benchmark Aspera settings and recommend the best (sequential with ETA).")
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--dest", required=True, help="Destination directory on server (docroot-relative or absolute if allowed)")
    ap.add_argument("--src",  required=True, help="Local source directory")
    ap.add_argument("--ssh-key", default=None)

    ap.add_argument("--rates", default="200,500,1000,2000", help="Comma-separated Mbps (e.g. 200,500,1000)")
    ap.add_argument("--policies", default="fair,high", help="Comma-separated (fair,high)")
    ap.add_argument("--sessions", default="1,3,5", help="Comma-separated session counts for classic ascp")
    ap.add_argument("--thresholds-mb", default="0,64,512", help="Comma-separated split thresholds in MB (0 disables splitting)")
    ap.add_argument("--ascp4", action="store_true", help="Include ascp4 tests (recommended for many-small files)")
    ap.add_argument("--extra-ascp-flags", default="", help="Extra flags to append to every ascp command (quoted string)")
    ap.add_argument("--extra-ascp4-flags", default="", help="Extra flags to append to every ascp4 command (quoted string)")

    ap.add_argument("--udp-base-port", type=int, default=33001, help="Base UDP port for multi-session; uses base..base+N-1")
    ap.add_argument("--isolate-dest", action="store_true", help="Put each test into its own subfolder under --dest")

    ap.add_argument("--sample-files", type=int, default=0, help="Max files to include (0 = all)")
    ap.add_argument("--sample-bytes", type=str, default="0", help="Approx bytes to include (e.g. 50G, 500M, 0=all)")

    ap.add_argument("--timeout-seconds", type=int, default=0, help="Per-test timeout; 0 = no timeout")
    ap.add_argument("--reps", type=int, default=1, help="Repeat each config N times (median throughput is used for ranking)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling and test order (default: random)")

    ap.add_argument("--log-dir", default="", help="Directory for logs; default creates ./aspera_bench_<timestamp>/")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, don’t execute")
    ap.add_argument("--json", action="store_true", help="Also emit JSON of the results to stdout at the end")

    args = ap.parse_args()

    # Validate environment early
    if which("ascp") is None:
        print("ERR: 'ascp' not found in PATH", file=sys.stderr); sys.exit(2)
    if args.ascp4 and which("ascp4") is None:
        print("ERR: '--ascp4' requested but 'ascp4' not found in PATH", file=sys.stderr); sys.exit(2)
    if args.ssh_key and not Path(args.ssh_key).expanduser().exists():
        print(f"ERR: SSH key not found: {args.ssh_key}", file=sys.stderr); sys.exit(2)

    # Parse sizes like "50G"
    def parse_size(s):
        s = s.strip().upper()
        if s == "0": return 0
        mult = 1
        if s.endswith("K"): mult=1024; s=s[:-1]
        elif s.endswith("M"): mult=1024**2; s=s[:-1]
        elif s.endswith("G"): mult=1024**3; s=s[:-1]
        elif s.endswith("T"): mult=1024**4; s=s[:-1]
        return int(float(s)*mult)
    max_bytes = parse_size(args.sample_bytes)

    # Logs root
    ts_root = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.log_dir) if args.log_dir else Path(f"./aspera_bench_{ts_root}")
    root_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    run_log_path = root_dir / "run.log"

    # Seed (reproducibility)
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)

    # Collect files
    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        print(f"ERR: Source {src} not found", file=sys.stderr); sys.exit(2)
    all_files = list_files(str(src))
    if not all_files:
        print("ERR: Source folder contains no files", file=sys.stderr); sys.exit(2)

    sample = pick_sample(all_files,
                         max_files=args.sample_files if args.sample_files>0 else None,
                         max_bytes=max_bytes if max_bytes>0 else None)
    sample_bytes = sum(sz for _,sz in sample)
    sample_count = len(sample)
    if sample_count == 0:
        print("ERR: Sample is empty after filtering", file=sys.stderr); sys.exit(2)

    # temp dir with cleanup
    tmpdir = tempfile.mkdtemp(prefix="aspera_bench_")
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    filelist_path = os.path.join(tmpdir, "filelist.txt")
    write_filelist(sample, filelist_path)

    rates      = [int(x) for x in args.rates.split(",") if x.strip()!=""]
    policies   = [x.strip() for x in args.policies.split(",") if x.strip()!=""]
    sessions   = [int(x) for x in args.sessions.split(",") if x.strip()!=""]
    thresholds = [int(x) for x in args.thresholds_mb.split(",") if x.strip()!=""]

    # Build test matrix
    tests=[]
    for r in rates:
        for pol in policies:
            tests.append(dict(tool="ascp", rate=r, policy=pol, sessions=1, threshold=None))
    for r in rates:
        for pol in policies:
            for ss in sessions:
                if ss <= 1: continue
                for thr in thresholds:
                    tests.append(dict(tool="ascp", rate=r, policy=pol, sessions=ss, threshold=thr))
    if args.ascp4:
        for r in rates:
            for pol in policies:
                tests.append(dict(tool="ascp4", rate=r, policy=pol, sessions=1, threshold=None))

    # Shuffle to reduce drift bias
    random.shuffle(tests)

    total_tests = len(tests)
    header = (f"[bench] Dataset: {sample_count} files, {hum_bytes(sample_bytes)} from {src}\n"
              f"[bench] Tests to run: {total_tests} combinations x reps={args.reps}\n"
              f"[bench] Random seed: {args.seed}\n"
              f"[bench] Logs: {root_dir}\n"
              + ("[bench] Each test uses its own destination subfolder.\n" if args.isolate_dest else ""))
    print("\n" + header)
    with open(run_log_path, "w", encoding="utf-8") as rlog:
        rlog.write(header)

    # Prepare CSV/MD paths inside the root
    ts = ts_root
    csv_path = root_dir / f"aspera_benchmark_results_{ts}.csv"
    md_path  = root_dir / f"aspera_benchmark_report_{ts}.md"

    # CSV now captures aggregate stats per CONFIG (after reps)
    csv_cols = [
        "rank","tool","policy","rate_mbps","sessions","split_threshold_mb",
        "rc_any_failure","rep_count",
        "seconds_median","seconds_mean","seconds_stdev",
        "bytes_median","bps_median","throughput_human_median",
        "dest_used","example_cmd","notes","hints"
    ]
    csv_f = open(csv_path, "w", newline='', encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(csv_cols)

    results=[]
    total_elapsed = 0.0
    completed = 0
    interrupted = False

    try:
        # Run tests sequentially (configs)
        for idx,conf in enumerate(tests,1):
            tool = conf["tool"]
            rate = conf["rate"]
            pol  = conf["policy"]
            sess = conf["sessions"]
            thr  = conf["threshold"]

            label_parts = [tool, pol, f"{rate}Mb", f"sess{sess}"]
            if thr is not None:
                label_parts.append(f"split{thr}MB")
            label = "_".join(label_parts)
            subdest = args.dest
            if args.isolate_dest:
                subdest = os.path.join(args.dest.rstrip("/"), slugify(label))

            # Build command(s) (template)
            extra = shlex.split(args.extra_ascp4_flags if tool=="ascp4" else args.extra_ascp_flags) if (args.extra_ascp4_flags or args.extra_ascp_flags) else []

            def build_cmds():
                cmds=[]
                notes=[]
                if tool=="ascp" and sess>1:
                    for i in range(1, sess+1):
                        cmd = build_ascp_cmd(
                            tool="ascp",
                            dest_user=args.user, dest_host=args.host, dest_path=subdest,
                            ssh_key=args.ssh_key, rate_mbps=rate, policy=pol, filelist_path=filelist_path,
                            session_id=i, session_count=sess, udp_port=args.udp_base_port + (i-1),
                            split_threshold_mb=thr, extra_flags=extra
                        )
                        cmds.append(cmd)
                    notes.append(f"ports {args.udp_base_port}-{args.udp_base_port+sess-1}")
                else:
                    cmd = build_ascp_cmd(
                        tool=tool,
                        dest_user=args.user, dest_host=args.host, dest_path=subdest,
                        ssh_key=args.ssh_key, rate_mbps=rate, policy=pol, filelist_path=filelist_path,
                        session_id=None, session_count=None, udp_port=None,
                        split_threshold_mb=thr if tool=="ascp" else None, extra_flags=extra
                    )
                    cmds.append(cmd)
                return cmds, notes

            # ETA calc (based on average config duration so far)
            avg = (total_elapsed / completed) if completed else None
            if avg:
                remaining = (total_tests - completed) * avg * max(args.reps, 1)
                print(f"[{idx}/{total_tests}] {label} -> {subdest}  | ETA ~ {hum_time(remaining)}")
            else:
                print(f"[{idx}/{total_tests}] {label} -> {subdest}")

            if args.dry_run:
                cmds, _notes = build_cmds()
                for c in cmds:
                    print("DRY:", " ".join(shlex.quote(x) for x in c))
                # still create a minimal per-test log for traceability
                tlog = logs_dir / f"{idx:03d}_{slugify(label)}.log"
                with open(tlog, "w", encoding="utf-8") as lf:
                    lf.write("[dry-run]\n")
                    for c in cmds:
                        lf.write("[cmd] " + " ".join(shlex.quote(x) for x in c) + "\n")
                continue

            # Perform repetitions and aggregate
            rep_seconds=[]; rep_bytes=[]; rep_bps=[]; any_fail=False
            hints_union=set(); notes_union=set()
            example_cmd=""

            for rep in range(1, args.reps+1):
                cmds, notes = build_cmds()
                notes_union.update(notes)
                if not example_cmd and cmds:
                    example_cmd = " ".join(shlex.quote(x) for x in cmds[0])

                tlog = logs_dir / f"{idx:03d}_{slugify(label)}_rep{rep}.log"
                rc, out, elapsed = run_parallel(cmds, str(tlog), timeout=args.timeout_seconds if args.timeout_seconds>0 else None)
                total_elapsed += elapsed
                # collect metrics
                bytes_tx = parse_bytes_transferred(out) if rc==0 else 0
                bps = (bytes_tx/elapsed) if (elapsed>0 and bytes_tx>0) else 0.0
                rep_seconds.append(elapsed); rep_bytes.append(bytes_tx); rep_bps.append(bps)
                any_fail = any_fail or (rc != 0)

                # Hints
                low = out.lower()
                if "http" in low and "fallback" in low:
                    hints_union.add("http_fallback_detected")
                if "udp" in low and "block" in low:
                    hints_union.add("udp_blocking_suspected")

                print(f"    rep {rep}/{args.reps} rc={rc} time={elapsed:.1f}s throughput~{hum_rate(bps)}  | log: {tlog.name}")

            completed += 1

            # Aggregate for this config
            sec_med = median(rep_seconds)
            sec_mean = mean(rep_seconds)
            sec_std = pstdev(rep_seconds) if len(rep_seconds) > 1 else 0.0
            bytes_med = median(rep_bytes)
            bps_med = median(rep_bps)

            results.append(dict(
                label=label, tool=tool, policy=pol, rate_mbps=rate, sessions=sess, split_mb=(thr if thr is not None else ""),
                rc_any_failure=int(any_fail), rep_count=len(rep_seconds),
                seconds_median=sec_med, seconds_mean=sec_mean, seconds_stdev=sec_std,
                bytes_median=bytes_med, bps_median=bps_med,
                dest_used=subdest, hints=";".join(sorted(hints_union)),
                example_cmd=example_cmd,
                notes=";".join(sorted(notes_union)),
                logs_glob=f"{idx:03d}_{slugify(label)}_rep*.log"
            ))

    except KeyboardInterrupt:
        interrupted = True
        print("\n[bench] Interrupted by user. Writing partial results...")

    finally:
        # Rank & write outputs (if any results collected)
        ranked=[]
        if results:
            # Rank by median throughput desc; tie-breaker: fewer sessions first (to prefer simpler configs)
            ranked = sorted(results, key=lambda r: (r["bps_median"], -r["sessions"]), reverse=True)
            # Prepare CSV writer
            for i,r in enumerate(ranked, start=1):
                csv_w.writerow([
                    i, r["tool"], r["policy"], r["rate_mbps"], r["sessions"], r["split_mb"],
                    r["rc_any_failure"], r["rep_count"],
                    f"{r['seconds_median']:.2f}", f"{r['seconds_mean']:.2f}", f"{r['seconds_stdev']:.2f}",
                    int(r["bytes_median"]), f"{r['bps_median']:.2f}", hum_rate(r["bps_median"]),
                    r["dest_used"], r["example_cmd"], r["notes"], r["hints"]
                ])
            csv_f.flush()
        csv_f.close()

        if args.dry_run:
            print("\n[bench] dry-run complete.")
            sys.exit(0)

        if not results:
            print("\n[bench] No results collected.")
            sys.exit(1)

        successes = [r for r in ranked if r["bytes_median"] > 0 and r["rc_any_failure"] == 0]
        if not successes:
            print("\n[bench] All tests failed or transferred 0 bytes. Check connectivity (UDP 33001+, SSH auth) or try single-session without multi-session.")
            with open(run_log_path, "a", encoding="utf-8") as rlog:
                rlog.write("\nAll tests failed or 0-byte transfers.\n")
            if args.json:
                print(json.dumps({"sample_files":len(sample), "sample_bytes":sample_bytes, "results":ranked}, indent=2))
            sys.exit(1)

        best = successes[0]

        # Console summary table (top 12)
        headers = ["Rank","Tool","Policy","Rate(Mb/s)","Sess","Split(MB)","Reps","Med Time(s)","Med Throughput"]
        rows=[]
        for i,r in enumerate(ranked[:12], start=1):
            rows.append([i, r["tool"], r["policy"], r["rate_mbps"], r["sessions"], r["split_mb"],
                         r["rep_count"], f"{r['seconds_median']:.1f}", hum_rate(r["bps_median"])])
        print("\n=== Top Results (first 12) ===")
        print(make_table(rows, headers))

        # Markdown report
        md_path = Path(md_path)  # ensure Path
        md_lines=[]
        md_lines.append(f"# Aspera Benchmark Report ({datetime.utcnow().isoformat(timespec='seconds')}Z)")
        md_lines.append("")
        md_lines.append(f"- Sample: **{len(sample)} files**, **{hum_bytes(sample_bytes)}**")
        md_lines.append(f"- Total configs: **{len(tests)}**, repetitions per config: **{args.reps}**")
        md_lines.append(f"- Random seed: `{args.seed}`")
        md_lines.append(f"- Logs dir: `{logs_dir}`")
        md_lines.append("")
        md_lines.append("## Recommendation")
        md_lines.append("")
        best_split = f" | split>={best['split_mb']} MB" if (best['tool']=='ascp' and best['sessions']>1 and best['split_mb']) else ""
        md_lines.append(f"**Best combo:** `{best['tool']} | policy={best['policy']} | rate={best['rate_mbps']} Mb/s | sessions={best['sessions']}{best_split}`")
        md_lines.append(f"- Observed median throughput: **{hum_rate(best['bps_median'])}** (median wall time {best['seconds_median']:.1f}s across {best['rep_count']} reps)")
        if best.get('hints'):
            md_lines.append(f"- Hints: `{best['hints']}`")
        md_lines.append(f"- Example command:\n\n```bash\n{best['example_cmd']}\n```")
        md_lines.append(f"- Per-config logs match: `{best['logs_glob']}`")
        md_lines.append("")
        md_lines.append("## Full Results")
        md_lines.append("")
        md_lines.append("| Rank | Tool | Policy | Rate (Mb/s) | Sessions | Split (MB) | Reps | Med Time (s) | Med Throughput | Dest | Hints |")
        md_lines.append("|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---|:---|")
        for i,r in enumerate(ranked, start=1):
            md_lines.append(f"| {i} | {r['tool']} | {r['policy']} | {r['rate_mbps']} | {r['sessions']} | {r['split_mb'] or ''} | "
                            f"{r['rep_count']} | {r['seconds_median']:.1f} | {hum_rate(r['bps_median'])} | `{r['dest_used']}` | `{r['hints']}` |")
        with open(md_path, "w", encoding="utf-8") as fmd:
            fmd.write("\n".join(md_lines))

        # Master run log tail
        with open(run_log_path, "a", encoding="utf-8") as rlog:
            rlog.write(f"\nBest: {best['tool']} policy={best['policy']} rate={best['rate_mbps']}Mb/s "
                       f"sessions={best['sessions']}{(' split>='+str(best['split_mb'])+'MB') if (best['tool']=='ascp' and best['sessions']>1 and best['split_mb']) else ''}  "
                       f"throughput={hum_rate(best['bps_median'])}\n")
            rlog.write(f"CSV: {csv_path}\nMD: {md_path}\n")

        print(f"\n=== Recommendation ===")
        print(f"{best['tool']} | policy={best['policy']} | rate={best['rate_mbps']} Mb/s | sessions={best['sessions']}{best_split}")
        print(f"Throughput (median): {hum_rate(best['bps_median'])}  (median elapsed {best['seconds_median']:.1f}s over {best['rep_count']} reps)")
        print("Example command:")
        print("   " + best["example_cmd"])

        print(f"\nSaved: {csv_path}")
        print(f"Saved: {md_path}")
        print(f"Per-test logs: {logs_dir}")

        if args.json:
            print(json.dumps({
                "sample_files": len(sample), "sample_bytes": sample_bytes,
                "results": ranked, "best": best,
                "csv": str(csv_path), "markdown": str(md_path), "logs_dir": str(logs_dir)
            }, indent=2))

        if interrupted:
            sys.exit(130)  # Ctrl-C exit code
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()
