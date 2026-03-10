import React, { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Separator } from "@/components/ui/separator";
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell, BarChart, Bar, Legend } from "recharts";
import { ShieldCheck, TriangleAlert, CircleAlert, CheckCircle2, FileSearch, Gauge, BriefcaseBusiness } from "lucide-react";

/**
 * Management-Oriented Contamination Risk Report
 * -------------------------------------------------
 * This component is designed as a fixed-structure management report webpage.
 * Replace `sampleData` with your pipeline JSON output.
 *
 * Expected input shape:
 * {
 *   meta: { reportTitle, generatedAt, dataset, benchmark, notes },
 *   executiveSummary: { overallMessage, keyFindings: string[] },
 *   models: [
 *     {
 *       name,
 *       benchmarkPerformance,
 *       contaminationRisk,
 *       confidence,
 *       businessReliability,
 *       recommendation,
 *       recommendationType,
 *       signals: { lexical, semantic, memorization, stability },
 *       riskDistribution: { low, medium, high, critical },
 *       highRiskShare,
 *       casesTotal
 *     }
 *   ],
 *   evidenceCases: [
 *     { id, model, risk, dominantEvidence, whyItMatters }
 *   ],
 *   auditability: [string],
 *   implications: [string],
 *   limitations: [string]
 * }
 */

const sampleData = {
  meta: {
    reportTitle: "Management-Oriented Contamination Risk Report",
    generatedAt: "2026-03-09",
    dataset: "XSum frozen evaluation set (296 cases)",
    benchmark: "Contamination assessment pipeline (SLex/SSem/SMem/SProb)",
    notes: "Values below are populated from current pipeline outputs in this repository.",
  },
  executiveSummary: {
    overallMessage:
      "Both compared models show elevated integrated contamination risk on this benchmark setup, with Gemini scoring higher semantic/memorization signals and GPT-4o-mini scoring higher instability.",
    keyFindings: [
      "Lexical exposure is high in the proxy corpus (SLex weighted index ~91.8/100), creating a high-exposure context for both models.",
      "GPT-4o-mini: CPS=0.392 (SSem level 1), EM_rate=0.0, UAR_mean=0.941, integrated RiskScore mean=53.30.",
      "Gemini-2.5-flash: CPS=0.500 (SSem level 2), EM_rate=0.0034, UAR_mean=0.694, integrated RiskScore mean=56.91.",
    ],
  },
  models: [
    {
      name: "gpt-4o-mini",
      benchmarkPerformance: 39.2,
      contaminationRisk: 53.3,
      confidence: 94.6,
      businessReliability: "Medium",
      recommendation: "Use with caution for benchmark-based claims; high-risk share remains substantial.",
      recommendationType: "pilot",
      signals: { lexical: 91.8, semantic: 33.3, memorization: 0.0, stability: 96.2 },
      riskDistribution: { low: 3.04, medium: 15.54, high: 81.42, critical: 0.0 },
      highRiskShare: 81.42,
      casesTotal: 296,
    },
    {
      name: "gemini-2.5-flash",
      benchmarkPerformance: 50.0,
      contaminationRisk: 56.9,
      confidence: 97.0,
      businessReliability: "Low",
      recommendation: "Highest integrated risk among compared models; require stronger out-of-benchmark validation.",
      recommendationType: "avoid",
      signals: { lexical: 91.8, semantic: 66.7, memorization: 3.9, stability: 67.8 },
      riskDistribution: { low: 1.35, medium: 8.78, high: 89.19, critical: 0.68 },
      highRiskShare: 89.87,
      casesTotal: 296,
    },
  ],
  evidenceCases: [
    {
      id: "xsum_id:35059626",
      model: "gemini-2.5-flash",
      risk: "Critical",
      dominantEvidence: "SLex=3, SSem=2, SMem=3, SProb=3 (RiskScore=93.33)",
      whyItMatters: "Critical-level integrated signal across all detectors indicates the benchmark case is highly exposure-sensitive.",
    },
    {
      id: "xsum_id:35082705",
      model: "gemini-2.5-flash",
      risk: "Critical",
      dominantEvidence: "SLex=3, SSem=2, SMem=2, SProb=2 (RiskScore=78.33)",
      whyItMatters: "Independent evidence channels align at high levels, reinforcing contamination-risk concern for this item.",
    },
    {
      id: "xsum_id:37255399",
      model: "gpt-4o-mini",
      risk: "High",
      dominantEvidence: "SLex=3 with high instability SProb=3 (RiskScore=56.67)",
      whyItMatters: "Even with low memorization score, high lexical exposure plus instability keeps case-level risk elevated.",
    },
  ],
  auditability: [
    "Frozen dataset: data/master_table_xsum_n300_seed42_v2_dcq_frozen_FINAL.parquet",
    "Versioned run artifacts: runs/v3..v7_* with model-specific logs",
    "Fixed prompts and deterministic settings for DCQ/Mem stages",
    "Traceable risk integration with explicit weights and override rules",
  ],
  implications: [
    "Benchmark scores should be interpreted jointly with contamination-risk diagnostics, not in isolation.",
    "High lexical overlap context (SLex-heavy) can inflate confidence in apparent model capability.",
    "Selection decisions should be re-validated on fresh, domain-specific holdouts with low exposure risk.",
  ],
  limitations: [
    "Contamination risk is inferred from observable signals and does not directly prove pretraining exposure.",
    "A low risk estimate should not be interpreted as proof of complete data cleanliness.",
    "Black-box assessment supports decision-making under uncertainty rather than causal attribution.",
  ],
};

const riskColorClass = {
  Low: "bg-emerald-100 text-emerald-800",
  Medium: "bg-amber-100 text-amber-800",
  High: "bg-orange-100 text-orange-800",
  Critical: "bg-rose-100 text-rose-800",
};

const recBadgeClass = {
  preferred: "bg-emerald-100 text-emerald-800",
  pilot: "bg-amber-100 text-amber-800",
  avoid: "bg-rose-100 text-rose-800",
};

function scoreToLabel(score) {
  if (score >= 75) return "High";
  if (score >= 50) return "Medium";
  return "Low";
}

function RecommendationIcon({ type }) {
  if (type === "preferred") return <CheckCircle2 className="h-4 w-4" />;
  if (type === "pilot") return <CircleAlert className="h-4 w-4" />;
  return <TriangleAlert className="h-4 w-4" />;
}

function MetricCard({ title, value, description, icon: Icon }) {
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-sm text-slate-500">{title}</p>
            <p className="mt-2 text-3xl font-semibold tracking-tight">{value}</p>
            <p className="mt-2 text-sm text-slate-600">{description}</p>
          </div>
          <div className="rounded-2xl bg-slate-100 p-3">
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SectionTitle({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-1 text-sm text-slate-600">{description}</p>
    </div>
  );
}

export default function ManagementContaminationReport() {
  const data = sampleData;

  const preferredModel = useMemo(() => {
    return [...data.models].sort((a, b) => {
      const aScore = a.benchmarkPerformance - a.contaminationRisk * 0.7;
      const bScore = b.benchmarkPerformance - b.contaminationRisk * 0.7;
      return bScore - aScore;
    })[0];
  }, [data.models]);

  const riskScatterData = data.models.map((m) => ({
    name: m.name,
    performance: m.benchmarkPerformance,
    risk: m.contaminationRisk,
    confidence: m.confidence,
  }));

  const distributionData = data.models.map((m) => ({
    name: m.name,
    Low: m.riskDistribution.low,
    Medium: m.riskDistribution.medium,
    High: m.riskDistribution.high,
    Critical: m.riskDistribution.critical,
  }));

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        <div className="grid gap-6">
          <Card className="rounded-3xl border-0 shadow-sm">
            <CardContent className="p-8">
              <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                <div className="max-w-3xl">
                  <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-700">
                    <BriefcaseBusiness className="h-4 w-4" />
                    Management Report
                  </div>
                  <h1 className="text-4xl font-semibold tracking-tight">{data.meta.reportTitle}</h1>
                  <p className="mt-3 text-base leading-7 text-slate-600">
                    This report translates contamination assessment outputs into a decision-oriented view for model selection, procurement, and deployment governance.
                  </p>
                  <div className="mt-4 grid gap-2 text-sm text-slate-600 sm:grid-cols-2">
                    <p><span className="font-medium text-slate-800">Generated:</span> {data.meta.generatedAt}</p>
                    <p><span className="font-medium text-slate-800">Dataset:</span> {data.meta.dataset}</p>
                    <p><span className="font-medium text-slate-800">Benchmark:</span> {data.meta.benchmark}</p>
                    <p><span className="font-medium text-slate-800">Notes:</span> {data.meta.notes}</p>
                  </div>
                </div>
                <Card className="w-full max-w-md rounded-3xl bg-slate-900 text-white shadow-none">
                  <CardContent className="p-6">
                    <p className="text-sm text-slate-300">Preferred candidate</p>
                    <h2 className="mt-2 text-2xl font-semibold">{preferredModel.name}</h2>
                    <p className="mt-3 text-sm leading-6 text-slate-300">
                      {preferredModel.recommendation}
                    </p>
                    <div className="mt-4 flex flex-wrap gap-2">
                      <Badge className="border-0 bg-white/10 text-white">Performance {preferredModel.benchmarkPerformance}</Badge>
                      <Badge className="border-0 bg-white/10 text-white">Risk {preferredModel.contaminationRisk}</Badge>
                      <Badge className="border-0 bg-white/10 text-white">Confidence {preferredModel.confidence}</Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-3">
            <MetricCard
              title="Decision focus"
              value="Trust benchmark with context"
              description="Performance should be interpreted together with contamination risk and confidence in the assessment."
              icon={Gauge}
            />
            <MetricCard
              title="Core question"
              value="Will performance transfer?"
              description="The report helps distinguish strong models from models that may be overestimated by benchmark contamination."
              icon={ShieldCheck}
            />
            <MetricCard
              title="Evidence standard"
              value="Auditable"
              description="Recommendations are supported by structured signals, illustrative cases, and a reproducible scoring pipeline."
              icon={FileSearch}
            />
          </div>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-2xl">4.X.1 Executive decision summary</CardTitle>
              <CardDescription>
                A concise management view of the main findings and recommended model-selection posture.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Alert className="rounded-2xl border-slate-200 bg-white">
                <BriefcaseBusiness className="h-4 w-4" />
                <AlertTitle>Overall message</AlertTitle>
                <AlertDescription className="mt-2 leading-7">
                  {data.executiveSummary.overallMessage}
                </AlertDescription>
              </Alert>

              <div className="grid gap-3 md:grid-cols-3">
                {data.executiveSummary.keyFindings.map((item, idx) => (
                  <Card key={idx} className="rounded-2xl">
                    <CardContent className="p-5 text-sm leading-6 text-slate-700">
                      {item}
                    </CardContent>
                  </Card>
                ))}
              </div>

              <div className="rounded-2xl border bg-white">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Model</TableHead>
                      <TableHead>Benchmark Performance</TableHead>
                      <TableHead>Contamination Risk</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Business Reliability</TableHead>
                      <TableHead>Recommendation</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.models.map((m) => (
                      <TableRow key={m.name}>
                        <TableCell className="font-medium">{m.name}</TableCell>
                        <TableCell>{m.benchmarkPerformance}</TableCell>
                        <TableCell>{m.contaminationRisk} ({scoreToLabel(m.contaminationRisk)})</TableCell>
                        <TableCell>{m.confidence}</TableCell>
                        <TableCell>{m.businessReliability}</TableCell>
                        <TableCell>
                          <Badge className={`border-0 ${recBadgeClass[m.recommendationType]} inline-flex items-center gap-1`}>
                            <RecommendationIcon type={m.recommendationType} />
                            {m.recommendationType === "preferred" ? "Preferred" : m.recommendationType === "pilot" ? "Pilot only" : "Avoid reliance"}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-2xl">4.X.2 Comparative model risk overview</CardTitle>
              <CardDescription>
                This view compares benchmark strength against contamination risk to support procurement-oriented interpretation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[360px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="performance" name="Performance" domain={[0, 100]} />
                    <YAxis type="number" dataKey="risk" name="Risk" domain={[0, 100]} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value, name) => [value, name]} />
                    <Scatter data={riskScatterData}>
                      {riskScatterData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.risk >= 75 ? "#ef4444" : entry.risk >= 50 ? "#f59e0b" : "#10b981"}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-600">
                Models positioned further to the right offer stronger benchmark performance, while models higher on the chart present greater contamination risk. The most attractive procurement candidates are those with strong performance and lower contamination risk.
              </p>
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-2xl">4.X.3 Contamination risk profile by model</CardTitle>
              <CardDescription>
                Signal-level profiles help explain why a model was assessed as lower or higher risk.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 lg:grid-cols-3">
                {data.models.map((m) => (
                  <Card key={m.name} className="rounded-2xl border-slate-200">
                    <CardHeader>
                      <CardTitle className="text-lg">{m.name}</CardTitle>
                      <CardDescription>{m.recommendation}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {Object.entries(m.signals).map(([k, v]) => (
                        <div key={k}>
                          <div className="mb-2 flex items-center justify-between text-sm">
                            <span className="capitalize">{k}</span>
                            <span className="font-medium">{v}</span>
                          </div>
                          <Progress value={v} className="h-2" />
                        </div>
                      ))}
                      <Separator />
                      <div className="flex items-center justify-between text-sm">
                        <span>Integrated contamination risk</span>
                        <span className="font-semibold">{m.contaminationRisk}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span>Confidence in assessment</span>
                        <span className="font-semibold">{m.confidence}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-2xl">4.X.4 Distribution of risk across evaluation cases</CardTitle>
              <CardDescription>
                Distribution matters because average risk alone does not show whether contamination is isolated or systematic.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[380px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={distributionData} layout="vertical" margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis dataKey="name" type="category" width={90} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Low" stackId="a" fill="#10b981" />
                    <Bar dataKey="Medium" stackId="a" fill="#f59e0b" />
                    <Bar dataKey="High" stackId="a" fill="#f97316" />
                    <Bar dataKey="Critical" stackId="a" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {data.models.map((m) => (
                  <Card key={m.name} className="rounded-2xl">
                    <CardContent className="p-5">
                      <p className="text-sm text-slate-500">{m.name}</p>
                      <p className="mt-2 text-2xl font-semibold">{m.highRiskShare}%</p>
                      <p className="mt-1 text-sm text-slate-600">High + critical share of evaluated cases</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="confidence" className="w-full">
            <TabsList className="grid w-full grid-cols-3 rounded-2xl">
              <TabsTrigger value="confidence">Confidence</TabsTrigger>
              <TabsTrigger value="evidence">Evidence</TabsTrigger>
              <TabsTrigger value="governance">Governance</TabsTrigger>
            </TabsList>

            <TabsContent value="confidence" className="mt-4">
              <Card className="rounded-3xl shadow-sm">
                <CardHeader>
                  <CardTitle className="text-2xl">4.X.5 Confidence and uncertainty in the assessment</CardTitle>
                  <CardDescription>
                    Confidence indicates how strongly the available evidence supports the assigned risk level.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-3">
                    {data.models.map((m) => (
                      <Card key={m.name} className="rounded-2xl">
                        <CardContent className="p-5 space-y-3">
                          <div className="flex items-center justify-between">
                            <p className="font-medium">{m.name}</p>
                            <Badge className="border-0 bg-slate-100 text-slate-800">Confidence {m.confidence}</Badge>
                          </div>
                          <p className="text-sm text-slate-600 leading-6">
                            Risk level: <span className="font-medium">{scoreToLabel(m.contaminationRisk)}</span>. Confidence should be interpreted together with signal convergence and the strength of illustrative evidence.
                          </p>
                          <Progress value={m.confidence} className="h-2" />
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="evidence" className="mt-4">
              <Card className="rounded-3xl shadow-sm">
                <CardHeader>
                  <CardTitle className="text-2xl">4.X.6 Evidence highlights: illustrative high-risk cases</CardTitle>
                  <CardDescription>
                    Representative cases help make the assessment concrete and auditable for non-technical decision-makers.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 lg:grid-cols-3">
                    {data.evidenceCases.map((c) => (
                      <Card key={c.id} className="rounded-2xl border-slate-200">
                        <CardContent className="p-5">
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <p className="font-medium">{c.id}</p>
                              <p className="text-sm text-slate-500">{c.model}</p>
                            </div>
                            <Badge className={`border-0 ${riskColorClass[c.risk]}`}>{c.risk}</Badge>
                          </div>
                          <p className="mt-4 text-sm font-medium text-slate-800">Dominant evidence</p>
                          <p className="mt-1 text-sm leading-6 text-slate-600">{c.dominantEvidence}</p>
                          <p className="mt-4 text-sm font-medium text-slate-800">Why it matters</p>
                          <p className="mt-1 text-sm leading-6 text-slate-600">{c.whyItMatters}</p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="governance" className="mt-4">
              <div className="grid gap-4 lg:grid-cols-2">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle className="text-2xl">4.X.7 Auditability and reproducibility</CardTitle>
                    <CardDescription>
                      These controls help explain why the resulting recommendations are suitable for governance and procurement review.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {data.auditability.map((item, idx) => (
                      <div key={idx} className="rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-700">
                        {item}
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle className="text-2xl">4.X.8 Business implications and recommended actions</CardTitle>
                    <CardDescription>
                      This section translates the findings into concrete procurement and validation guidance.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div>
                      <p className="mb-3 text-sm font-medium text-slate-800">Implications</p>
                      <div className="space-y-3">
                        {data.implications.map((item, idx) => (
                          <div key={idx} className="rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-700">
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>
                    <Separator />
                    <div>
                      <p className="mb-3 text-sm font-medium text-slate-800">Limits of managerial interpretation</p>
                      <div className="space-y-3">
                        {data.limitations.map((item, idx) => (
                          <div key={idx} className="rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-700">
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
