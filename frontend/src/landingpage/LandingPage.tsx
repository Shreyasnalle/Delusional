"use client";

import React, { useState, useEffect, useRef } from "react";

function GithubIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

function LinkedinIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.46 10.9v8.37H9.25V10.9H6.46M7.86 6.74a1.64 1.64 0 0 0-1.64 1.64c0 .9.73 1.64 1.64 1.64a1.64 1.64 0 0 0 1.64-1.64c0-.9-.73-1.64-1.64-1.64Z" />
    </svg>
  );
}

export default function LandingPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [currentStep, setCurrentStep] = useState<"idle" | "attacking" | "defending" | "finetuning" | "complete">("idle");
  const [hasRun, setHasRun] = useState(false);
  const [modelCount, setModelCount] = useState(1);

  const [attackMetrics, setAttackMetrics] = useState({
    totalTransactions: "100,000",
    totalFrauds: "39,790 (39.8%)",
    totalNormal: "60,210 (60.2%)",
    attackFilename: "ATTACK 1"
  });

  const [defendStatus, setDefendStatus] = useState<"Standby" | "Defending..." | "Defending Done">("Standby");

  const [modelMetrics, setModelMetrics] = useState({
    savePath: "Fine Tuned Model 1",
    totalTransactions: 100000,
    totalRealTransactions: 60210,
    totalTrueFrauds: 39790,
    beforeDetected: "28,086",
    beforeRate: "70.59%",
    beforeTp: 28086,
    beforeFn: 11704,
    beforeTn: 45498,
    beforeFp: 14712,
    beforePrecision: "0.6562",
    beforeRecall: "70.59%",
    afterDetected: "38,874",
    afterRate: "97.70%",
    afterTp: 38874,
    afterFn: 916,
    afterTn: 53342,
    afterFp: 6868,
    afterPrecision: "0.8499",
    afterRecall: "97.70%",
    precisionPct: "84.99%",
    performancePct: "97.70%"
  });

  const card1Ref = useRef<HTMLDivElement>(null);
  const card3Ref = useRef<HTMLDivElement>(null);
  const arrowAnchorRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);

  const [pathD, setPathD] = useState("");
  const [arrowHeadPos, setArrowHeadPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const updatePath = () => {
      if (!card1Ref.current || !card3Ref.current || !mainRef.current || !arrowAnchorRef.current) return;

      const mainRect = mainRef.current.getBoundingClientRect();
      const card1Rect = card1Ref.current.getBoundingClientRect();
      const card3Rect = card3Ref.current.getBoundingClientRect();
      const anchorRect = arrowAnchorRef.current.getBoundingClientRect();

      const startX = (card3Rect.left + card3Rect.width / 2) - mainRect.left;
      const startY = (card3Rect.bottom - mainRect.top) + 22;

      const endX = (card1Rect.left + card1Rect.width / 2) - mainRect.left;
      const endY = (card1Rect.bottom - mainRect.top) + 24;

      const lineY = (anchorRect.bottom + 12) - mainRect.top;
      const cornerRadius = 24;

      const d = `
        M ${startX} ${startY}
        V ${lineY - cornerRadius}
        Q ${startX} ${lineY} ${startX - cornerRadius} ${lineY}
        H ${endX + cornerRadius}
        Q ${endX} ${lineY} ${endX} ${lineY - cornerRadius}
        V ${endY + 6}
      `;

      setPathD(d);
      setArrowHeadPos({ x: endX, y: endY + 6 });
    };

    updatePath();
    window.addEventListener("resize", updatePath);
    const timer = setTimeout(updatePath, 200);
    return () => {
      window.removeEventListener("resize", updatePath);
      clearTimeout(timer);
    };
  }, []);

  const handleStartSimulation = () => {
    if (isRunning) return;
    setIsRunning(true);
    setCurrentStep("attacking");
    setDefendStatus("Standby");

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const eventSource = new EventSource(`${backendUrl}/api/run-pipeline-stream`);

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.step === "attacking") {
            setCurrentStep("attacking");
          } else if (data.step === "attack_complete") {
            const rawFile = data.attack_filename || `attack_${modelCount}.csv`;
            const formattedName = rawFile.replace(/attack_(\d+)\.csv/i, "ATTACK $1").replace(/_/g, " ").replace(/\.csv/i, "").toUpperCase();
            setAttackMetrics({
              totalTransactions: data.total_transactions || "100,000",
              totalFrauds: data.total_frauds || "39,790 (39.8%)",
              totalNormal: data.total_normal || "60,210 (60.2%)",
              attackFilename: formattedName
            });
          } else if (data.step === "defending") {
            setCurrentStep("defending");
            setDefendStatus("Defending...");
          } else if (data.step === "defending_complete") {
            setDefendStatus("Defending Done");
          } else if (data.step === "finetuning") {
            setCurrentStep("finetuning");
          } else if (data.step === "complete") {
            eventSource.close();
            if (data.metrics) {
              const m = data.metrics;
              const totalTx = m.total_transactions ?? (m.before ? m.before.tp + m.before.fn + m.before.tn + m.before.fp : 100000);
              const totalFraud = m.total_fraud_transactions ?? m.total_true_frauds ?? (m.before ? m.before.tp + m.before.fn : 39790);
              const totalReal = m.total_real_transactions ?? (totalTx - totalFraud);
              const precVal = m.after?.precision ?? 0.8499;
              const precPctStr = `${(precVal * 100).toFixed(2)}%`;
              const rawSavePath = m.save_path || `fine_tuned_model_${modelCount}.pt`;
              const matchNum = rawSavePath.match(/(\d+)/);
              const modelNum = matchNum ? matchNum[1] : modelCount;
              const formattedCheckpoint = `Fine Tuned Model ${modelNum}`;

              setModelMetrics({
                savePath: formattedCheckpoint,
                totalTransactions: totalTx,
                totalRealTransactions: totalReal,
                totalTrueFrauds: totalFraud,
                beforeDetected: m.before?.detected_str || "28,086",
                beforeRate: m.before?.rate_str || "70.59%",
                beforeTp: m.before?.tp ?? 28086,
                beforeFn: m.before?.fn ?? 11704,
                beforeTn: m.before?.tn ?? 45498,
                beforeFp: m.before?.fp ?? 14712,
                beforePrecision: (m.before?.precision ?? 0.6562).toFixed(4),
                beforeRecall: m.before?.rate_str || "70.59%",
                afterDetected: m.after?.detected_str || "38,874",
                afterRate: m.after?.rate_str || "97.70%",
                afterTp: m.after?.tp ?? 38874,
                afterFn: m.after?.fn ?? 916,
                afterTn: m.after?.tn ?? 53342,
                afterFp: m.after?.fp ?? 6868,
                afterPrecision: precVal.toFixed(4),
                afterRecall: m.after?.rate_str || "97.70%",
                precisionPct: precPctStr,
                performancePct: m.performance_pct || "97.70%"
              });
            }

            setHasRun(true);
            setModelCount(prev => prev + 1);
            setCurrentStep("complete");
            setIsRunning(false);
          }
        } catch (err) {
          console.error("Error parsing pipeline event:", err);
        }
      };

      eventSource.onerror = (err) => {
        console.warn("EventSource stream error, switching to timed simulation fallback:", err);
        eventSource.close();
        runFallbackSimulation();
      };
    } catch (e) {
      console.warn("API Connection unavailable, running fallback:", e);
      runFallbackSimulation();
    }
  };

  const runFallbackSimulation = () => {
    setTimeout(() => {
      setCurrentStep("defending");
      setDefendStatus("Defending...");
    }, 1500);

    setTimeout(() => {
      setDefendStatus("Defending Done");
    }, 2800);

    setTimeout(() => {
      setCurrentStep("finetuning");
    }, 3200);

    setTimeout(() => {
      setHasRun(true);
      setModelCount(prev => prev + 1);
      setCurrentStep("complete");
      setIsRunning(false);
    }, 5500);
  };

  const cardClass = "group relative rounded-xl p-5 transition-all duration-300 backdrop-blur-[20px] bg-[#0b1426]/95 border border-slate-700/60 shadow-xl flex flex-col justify-between h-[240px] w-full";

  return (
    <div className="relative h-screen max-h-screen w-full overflow-hidden bg-[#070e1b] font-['Vercetti',sans-serif] text-slate-100 selection:bg-sky-500 selection:text-white flex flex-col justify-between">
      <div 
        className="absolute inset-0 z-0 bg-cover bg-center bg-no-repeat pointer-events-none"
        style={{ backgroundImage: "url('/sky-bg.png')" }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-black/25 via-transparent to-black/40" />
      </div>

      <header className="relative z-20 w-full max-w-7xl mx-auto px-6 pt-8 md:pt-10 pb-2 flex items-center justify-center text-center">
        <div className="flex flex-col items-center justify-center text-center">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-wider text-white uppercase drop-shadow-[0_2px_10px_rgba(0,0,0,0.8)]">
            GENAI FRAUD DEFENSE
          </h1>
          <p className="text-xs md:text-sm text-sky-200/90 font-medium tracking-widest uppercase mt-0.5">
            Self improving multi gnn model for genai frauds
          </p>
        </div>
      </header>

      <main ref={mainRef} className="relative z-10 w-full max-w-6xl mx-auto px-4 py-2 flex-1 flex flex-col justify-around">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8 w-full items-stretch relative z-20">
          <div ref={card1Ref} className={cardClass}>
            <div>
              <div className="flex items-center justify-between mb-2 border-b border-slate-700/50 pb-2">
                <h2 className="text-base font-bold tracking-widest text-white uppercase">MODEL INSIGHTS</h2>
              </div>

              <div className="bg-[#060c18]/90 p-3 rounded-lg border border-slate-800 h-[115px] flex flex-col justify-center">
                {!hasRun ? (
                  <div className="text-center py-2">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      No stats, please run the model
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-300 uppercase tracking-wider">Before Fine-Tuning</span>
                      <span className="text-xs font-mono font-bold text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">
                        {modelMetrics.beforeDetected} ({modelMetrics.beforeRate})
                      </span>
                    </div>

                    <div className="flex items-center justify-between pt-1.5 border-t border-slate-800">
                      <span className="text-xs text-slate-100 uppercase tracking-wider font-bold">After Fine-Tuning</span>
                      <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/15 px-2 py-0.5 rounded border border-emerald-500/30">
                        {modelMetrics.afterDetected} ({modelMetrics.afterRate})
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="pt-2 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-300 relative">
              <span className="text-slate-400 uppercase tracking-wider">Status</span>
              
              <div className="group/precision relative flex items-center gap-1.5 cursor-help">
                <span className="font-mono text-sky-300 font-bold uppercase tracking-wider">
                  Precision: <strong className="text-emerald-400">{hasRun ? modelMetrics.precisionPct : "--%"}</strong>
                </span>
                <span className="text-[10px] font-bold text-sky-300/80 bg-sky-500/20 border border-sky-400/40 rounded-full w-4 h-4 flex items-center justify-center">
                  ?
                </span>

                <div className="absolute right-0 bottom-full mb-2 hidden group-hover/precision:flex flex-col w-72 p-3.5 rounded-xl bg-[#081021] border border-sky-400/60 shadow-[0_10px_35px_rgba(0,0,0,0.95)] text-xs z-[100] animate-fade-in backdrop-blur-2xl">
                  <div className="mb-1.5 border-b border-slate-700/80 pb-1.5">
                    <span className="font-bold text-sky-300 uppercase tracking-wider text-xs">WHAT IS PRECISION?</span>
                  </div>
                  <p className="text-[11px] leading-relaxed text-slate-100 normal-case font-normal">
                    When the GNN flags a transaction as fraud, this represents the exact percentage that are actual fraud vs false alarms on clean customers.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className={cardClass}>
            <div>
              <div className="flex items-center justify-between mb-2 border-b border-slate-700/50 pb-2">
                <h2 className="text-base font-bold tracking-widest text-white uppercase">ATTACK GENERATION</h2>
              </div>

              <div className="bg-[#060c18]/90 p-3 rounded-lg border border-slate-800 h-[115px] flex flex-col justify-center items-center text-center">
                <p className="text-xs font-bold text-slate-300 tracking-wider uppercase mb-1">
                  STATUS OF ATTACK GENERATION
                </p>

                {currentStep === "attacking" ? (
                  <div className="py-2 text-center text-xs font-bold text-rose-400 uppercase tracking-widest animate-pulse">
                    Generating attack...
                  </div>
                ) : (currentStep === "defending" || currentStep === "finetuning" || currentStep === "complete") ? (
                  <div className="space-y-1.5 text-xs font-mono w-full px-1">
                    <div className="flex items-center justify-between text-slate-300">
                      <span className="uppercase tracking-wider">Total Transactions:</span>
                      <strong className="text-white font-bold">{attackMetrics.totalTransactions}</strong>
                    </div>
                    <div className="flex items-center justify-between text-rose-300">
                      <span className="uppercase tracking-wider">Total Frauds:</span>
                      <strong className="text-rose-400 font-bold">{attackMetrics.totalFrauds}</strong>
                    </div>
                    <div className="flex items-center justify-between text-emerald-300">
                      <span className="uppercase tracking-wider">Total Normal Trans:</span>
                      <strong className="text-emerald-400 font-bold">{attackMetrics.totalNormal}</strong>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-400 uppercase tracking-wider text-center py-2">
                    Ready to generate attack dataset
                  </p>
                )}
              </div>
            </div>

            <div className="pt-2 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-400 uppercase tracking-wider">
              <span>Attack Engine</span>
              <span className="font-mono text-slate-200">{attackMetrics.attackFilename}</span>
            </div>
          </div>

          <div ref={card3Ref} className={cardClass}>
            <div>
              <div className="flex items-center justify-between mb-2 border-b border-slate-700/50 pb-2">
                <h2 className="text-base font-bold tracking-widest text-white uppercase">DEFEND</h2>
                <span className="text-[11px] font-bold uppercase tracking-widest px-2.5 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-400/30">
                  MULTI HETERO GNN SHIELD
                </span>
              </div>

              <div className="bg-[#060c18]/90 p-3 rounded-lg border border-slate-800 h-[115px] flex flex-col justify-center items-center text-center">
                <p className="text-xs font-bold text-slate-300 tracking-wider uppercase mb-1">
                  STATUS OF DEFEND
                </p>
                <p className="text-sm font-bold tracking-widest uppercase text-emerald-400">
                  {currentStep === "defending" ? (
                    <span className="text-sky-400 animate-pulse">Defending...</span>
                  ) : defendStatus === "Defending Done" || currentStep === "finetuning" || currentStep === "complete" ? (
                    <span className="text-emerald-400">Defending Done</span>
                  ) : (
                    <span className="text-slate-400 font-normal text-xs">Standby</span>
                  )}
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-400 uppercase tracking-wider">
              <span>Shield Status</span>
              <span className="font-mono text-sky-300">{currentStep === "complete" ? "Ready" : "Active"}</span>
            </div>
          </div>
        </div>

        <div className="my-3 flex flex-col items-center justify-center relative z-20">
          <button
            onClick={handleStartSimulation}
            disabled={isRunning}
            className="px-9 py-2.5 rounded-full font-bold text-white text-sm tracking-widest uppercase transition-transform duration-300 bg-[#0b1426]/95 hover:bg-[#111e38] border border-slate-700/60 shadow-xl hover:scale-105 active:scale-95 disabled:opacity-75 cursor-pointer z-20 mb-6"
          >
            {isRunning ? "EVALUATING..." : "START"}
          </button>

          <div ref={arrowAnchorRef} className="z-20 my-2">
            <span className="text-xs font-bold uppercase tracking-[0.25em] text-sky-200 bg-[#0b1426] px-5 py-2 rounded-full border border-sky-400/40 shadow-lg backdrop-blur-md">
              {currentStep === "finetuning" ? (
                <span className="text-amber-300 animate-pulse">Improving Model After Attack...</span>
              ) : (
                "Improving Model After Attack"
              )}
            </span>
          </div>

          <button
            onClick={() => setShowReport(true)}
            className="group px-6 py-2 rounded-full text-xs font-bold tracking-widest text-sky-200 hover:text-white uppercase transition-all duration-300 bg-[#0b1426]/90 hover:bg-slate-800 border border-sky-400/40 hover:border-sky-300 shadow-md hover:scale-105 cursor-pointer z-20 mt-6"
          >
            PERFORMANCE REPORT
          </button>
        </div>

        {pathD && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-10 overflow-visible">
            <defs>
              <linearGradient id="glow-line" x1="100%" y1="0%" x2="0%" y2="0%">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.9" />
                <stop offset="50%" stopColor="#818cf8" stopOpacity="1" />
                <stop offset="100%" stopColor="#34d399" stopOpacity="0.9" />
              </linearGradient>

              <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            <path
              d={pathD}
              fill="none"
              stroke="url(#glow-line)"
              strokeWidth="4"
              strokeDasharray="12 10"
              filter="url(#glow)"
              className="opacity-40"
            />

            <path
              d={pathD}
              fill="none"
              stroke="url(#glow-line)"
              strokeWidth="2.5"
              strokeDasharray="12 10"
              className={`${currentStep === "finetuning" ? "animate-dash-flow-fast" : "animate-dash-flow"} opacity-90`}
            />

            {arrowHeadPos.x > 0 && (
              <g transform={`translate(${arrowHeadPos.x}, ${arrowHeadPos.y}) rotate(0)`}>
                <path
                  d="M -6 8 L 0 -2 L 6 8 Z"
                  fill="#38bdf8"
                  filter="url(#glow)"
                />
              </g>
            )}
          </svg>
        )}
      </main>

      <footer 
        className="relative z-20 w-full max-w-7xl mx-auto px-6 py-3 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-400 font-['Vercetti',sans-serif]"
      >
        <div className="flex items-center gap-1.5">
          <span>Made with ❤️ by</span>
          <span className="font-semibold text-slate-200">Shreyas Nalle</span>
        </div>

        <div className="flex items-center gap-4 font-sans">
          <a 
            href="https://github.com/Shreyasnalle/Delusional" 
            target="_blank" 
            rel="noreferrer"
            className="flex items-center gap-1 hover:text-sky-300 transition-colors hover:underline"
          >
            <GithubIcon className="h-3.5 w-3.5" /> GitHub
          </a>
          <span className="text-slate-600">×</span>
          <a 
            href="https://www.linkedin.com/in/shreyas-nalle-0697bb371/" 
            target="_blank" 
            rel="noreferrer"
            className="flex items-center gap-1 hover:text-sky-300 transition-colors hover:underline"
          >
            <LinkedinIcon className="h-3.5 w-3.5" /> LinkedIn
          </a>
        </div>
      </footer>

      {showReport && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in font-['Vercetti',sans-serif]">
          <div className="relative w-full max-w-2xl rounded-xl bg-[#0b1426] border border-slate-700 p-6 md:p-8 shadow-2xl text-white">
            <div className="mb-4">
              <h3 className="text-xl font-bold uppercase tracking-wider text-white">BLUE TEAM DEFENSE COMPARISON REPORT</h3>
              <p className="text-xs text-sky-200/70 tracking-widest uppercase">Autonomous Fine-Tuning Performance</p>
            </div>

            {!hasRun ? (
              <div className="py-12 px-6 text-center my-6 bg-[#060c18]/90 rounded-lg border border-slate-800 space-y-2">
                <p className="text-base font-bold text-amber-400 uppercase tracking-widest">
                  No Performance Report Available Yet
                </p>
                <p className="text-xs text-slate-300 uppercase tracking-wider">
                  Please click START to generate an attack dataset and run model fine-tuning evaluation.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto my-4">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-xs font-bold text-slate-400 uppercase tracking-wider">
                      <th className="py-2.5 px-4">Metric</th>
                      <th className="py-2.5 px-4 text-center">Before Fine-Tuning</th>
                      <th className="py-2.5 px-4 text-center text-sky-300">After Fine-Tuning</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-xs uppercase tracking-wider">
                    <tr>
                      <td className="py-2.5 px-4 font-bold text-slate-300">Total Transactions</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.totalTransactions.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-white font-bold">{modelMetrics.totalTransactions.toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 font-bold text-slate-300">Total Real Transactions</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.totalRealTransactions.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-white font-bold">{modelMetrics.totalRealTransactions.toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 font-bold text-slate-300">Total Fraud Transactions</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.totalTrueFrauds.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-white font-bold">{modelMetrics.totalTrueFrauds.toLocaleString()}</td>
                    </tr>
                    <tr className="bg-emerald-500/5">
                      <td className="py-2.5 px-4 font-bold text-emerald-300">Frauds Detected (TP)</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.beforeTp.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-emerald-400 font-bold">
                        {modelMetrics.afterTp.toLocaleString()} (+{(modelMetrics.afterTp - modelMetrics.beforeTp).toLocaleString()})
                      </td>
                    </tr>
                    <tr className="bg-rose-500/5">
                      <td className="py-2.5 px-4 font-bold text-rose-300">Missed Frauds (FN)</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.beforeFn.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-rose-400 font-bold">
                        {modelMetrics.afterFn.toLocaleString()} (-{(modelMetrics.beforeFn - modelMetrics.afterFn).toLocaleString()})
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 font-bold text-slate-300">Real Trans Passed (TN)</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.beforeTn.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-white font-bold">{modelMetrics.afterTn.toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 font-bold text-slate-300">False Positives (FP)</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.beforeFp.toLocaleString()}</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.afterFp.toLocaleString()}</td>
                    </tr>
                    <tr className="border-t border-slate-700 bg-sky-500/10">
                      <td className="py-2.5 px-4 font-bold text-sky-300">Precision</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.beforePrecision}</td>
                      <td className="py-2.5 px-4 text-center text-sky-300 font-bold">{modelMetrics.afterPrecision}</td>
                    </tr>
                    <tr className="bg-sky-500/10">
                      <td className="py-2.5 px-4 font-bold text-sky-300">Recall (Fraud Catch Rate)</td>
                      <td className="py-2.5 px-4 text-center text-slate-400">{modelMetrics.beforeRecall}</td>
                      <td className="py-2.5 px-4 text-center text-emerald-400 font-bold text-sm">{modelMetrics.afterRecall}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            <div className="mt-6 pt-4 border-t border-slate-700 flex items-center justify-between text-xs uppercase tracking-wider">
              <span className="text-slate-400">
                {hasRun ? (
                  <>Checkpoint Model: <code className="text-sky-300 bg-slate-900 px-2 py-0.5 rounded">{modelMetrics.savePath}</code></>
                ) : (
                  "Status: Awaiting Initial Model Run"
                )}
              </span>
              <button 
                onClick={() => setShowReport(false)}
                className="px-5 py-2 rounded bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold transition-colors cursor-pointer uppercase tracking-wider"
              >
                Close Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
