const $ = (id) => document.getElementById(id);
const pages = ["strategy", "tests", "results"];
const state = {
    page: "strategy",
    overview: null,
    runId: "",
    run: null,
    poll: null,
    trainingPoll: null,
    trainingRequest: 0,
    trainingView: "",
    queueDeadline: 0,
    resultRequest: 0,
    estimateRequest: 0,
    starting: false,
};
const statuses = {
    queued: "排队中", pending: "等待测试", running: "测试中", cancelling: "正在取消",
    cancelled: "已取消", completed: "已完成", incomplete: "未完成", failed: "失败",
};
const trainingStatuses = { queued: "排队中", running: "训练中", completed: "已完成", failed: "失败", interrupted: "已中断" };
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 1 });
const timestamp = new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "medium" });
const scoreItems = [["kill", "击毁"], ["damage", "伤害"], ["survive", "生存"], ["time", "时间"]];

function element(tag, text, className = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    return node;
}

async function request(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json();
    if (!response.ok || payload.error) {
        throw new Error(payload.error || `工作台请求失败（${response.status}）`);
    }
    return payload;
}

async function command(...args) {
    const { data } = await request("/api/command", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ args }),
    });
    return data;
}

function showNotice(message, failed = false) {
    const notice = $("notice");
    notice.textContent = message;
    notice.classList.toggle("failure", failed);
    notice.hidden = !message;
}

async function withButton(button, callback) {
    button.disabled = true;
    showNotice("");
    try {
        await callback();
    } catch (error) {
        showNotice(error.message, true);
    } finally {
        button.disabled = false;
    }
}

function selectOptions(id, entries, value = $(id).value) {
    const select = $(id);
    const changed = select.options.length !== entries.length || entries.some(([key, name], index) => {
        const option = select.options[index];
        return option.value !== key || option.textContent !== name;
    });
    if (changed) select.replaceChildren(...entries.map(([key, name]) => new Option(name, key)));
    if (entries.some(([key]) => key === value)) select.value = value;
    select.disabled = entries.length === 0;
}

function versionLabel(id) {
    const version = state.overview.versions.find((item) => item.id === id);
    if (!version) return id.slice(0, 10);
    return id === "baseline" ? version.name : `${version.name} · ${id.slice(2, 10)}`;
}

function addRow(body, cells) {
    const row = document.createElement("tr");
    for (const value of cells) {
        const cell = document.createElement("td");
        if (value instanceof Node) cell.append(value);
        else cell.textContent = String(value);
        row.append(cell);
    }
    body.append(row);
}

function emptyTable(body, columns, text) {
    const cell = element("td", text, "muted");
    cell.colSpan = columns;
    const row = document.createElement("tr");
    row.append(cell);
    body.replaceChildren(row);
}

function renderVersions() {
    const { main, sources, versions } = state.overview;
    $("current-main").replaceChildren("主力版", element("strong", versionLabel(main)));
    const body = $("versions");
    body.replaceChildren();
    for (const item of versions) {
        const version = element("span", item.id === "baseline" ? "SDK" : item.id.slice(2, 10), "version-id");
        version.title = item.id;
        const action = item.id === main ? element("span", "主力版", "tag") : element("button", "设为主力");
        if (action instanceof HTMLButtonElement) {
            action.addEventListener("click", () => withButton(action, async () => {
                await command("snapshot", "main", item.id);
                await loadOverview(true);
                showNotice(`已将 ${versionLabel(item.id)} 设为主力版`);
            }));
        }
        addRow(body, [item.name, version, element("span", item.note || "—", "summary-col"), action]);
    }
    selectOptions("source", sources.map((source) => [source, source]));
    $("snapshot-form").querySelector("button").disabled = !sources.length;
    const onlineVersions = versions.filter((item) => item.id !== "baseline");
    selectOptions("online-version", onlineVersions.map(({ id }) => [id, versionLabel(id)]));
    for (const button of document.querySelectorAll("[data-online]")) button.disabled = !onlineVersions.length;
}

function selectedCandidates() {
    return [...$("candidates").querySelectorAll("input:checked")].map((chosen) => chosen.value);
}

function renderTestSetup(forceMain) {
    const { main, versions, suites } = state.overview;
    const allVersions = versions.map(({ id }) => [id, versionLabel(id)]);
    selectOptions("baseline", allVersions, forceMain ? main : $("baseline").value || main);
    selectOptions("suite", suites.map((suite) => [suite, suite]), $("suite").value || "quick");
    const selected = selectedCandidates();
    const choices = $("candidates");
    choices.replaceChildren();
    for (const [id, label] of allVersions) {
        if (id === $("baseline").value) continue;
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = id;
        checkbox.checked = selected.includes(id);
        const choice = element("label", "", "check");
        choice.append(checkbox, label);
        choices.append(choice);
    }
    if (!choices.childElementCount) choices.append(element("p", "请先创建策略快照", "muted"));
}

async function loadOverview(forceMain = false) {
    state.overview = await request("/api/overview");
    renderVersions();
    renderTestSetup(forceMain);
    const runs = state.overview.runs;
    if (!runs.some(({ run_id }) => run_id === state.runId)) state.runId = runs[0]?.run_id || "";
    selectOptions("run-select", runs.map((run) => [
        run.run_id, `${run.suite} · ${timestamp.format(new Date(run.created_at * 1000))}`,
    ]), state.runId);
    $("refresh-results").disabled = !state.runId;
    $("no-runs").hidden = Boolean(state.runId);
    $("run-content").hidden = !state.runId;
}

function testTrainingSnapshot(snapshotId) {
    const baseline = $("baseline");
    if (baseline.value === snapshotId) {
        baseline.value = "baseline";
        renderTestSetup(false);
    }
    const candidate = [...$("candidates").querySelectorAll("input")].find(({ value }) => value === snapshotId);
    candidate.checked = true;
    location.hash = "tests";
}

function renderTraining(jobs) {
    const view = JSON.stringify([jobs, state.overview.versions.map(({ id }) => id)]);
    if (view === state.trainingView) return;
    state.trainingView = view;
    $("training-panel").hidden = jobs.length === 0;
    const body = $("training-jobs");
    body.replaceChildren();
    for (const job of jobs) {
        const status = element("span", trainingStatuses[job.status] || job.status);
        if (job.error) status.append(element("span", job.error, "training-error"));
        let output = job.checkpoint_id ? "已登记检查点" : "—";
        if (job.snapshot_id && state.overview.versions.some(({ id }) => id === job.snapshot_id)) {
            const button = element("button", "测试此版本");
            button.addEventListener("click", () => testTrainingSnapshot(job.snapshot_id));
            output = element("div", versionLabel(job.snapshot_id), "training-output");
            output.append(button);
        }
        addRow(body, [job.script || "准备中", timestamp.format(new Date(job.created_at * 1000)), status, output]);
    }
}

async function refreshTraining() {
    clearTimeout(state.trainingPoll);
    const request = ++state.trainingRequest;
    try {
        const jobs = await command("jobs", "list");
        if (state.page !== "strategy" || request !== state.trainingRequest) return;
        if (jobs.some(({ snapshot_id }) => snapshot_id && !state.overview.versions.some(({ id }) => id === snapshot_id))) {
            await loadOverview();
        }
        if (state.page === "strategy" && request === state.trainingRequest) renderTraining(jobs);
    } catch (error) {
        if (state.page === "strategy" && request === state.trainingRequest) showNotice(error.message, true);
    } finally {
        if (state.page === "strategy" && request === state.trainingRequest) {
            state.trainingPoll = setTimeout(refreshTraining, 5000);
        }
    }
}

function testArgs(candidates, baseline, suite, mode) {
    return ["--candidates", candidates.join(","), "--baseline", baseline, "--suite", suite, "--mode", mode];
}

function currentTestArgs() {
    const args = testArgs(selectedCandidates(), $("baseline").value, $("suite").value, $("mode").value);
    if ($("save-replay").checked) args.push("--replay");
    return args;
}

function duration(seconds) {
    if (seconds === null) return "待有对局记录后估算";
    if (seconds < 60) return `约 ${Math.ceil(seconds)} 秒`;
    return `约 ${Math.ceil(seconds / 60)} 分钟`;
}

async function estimate() {
    const requestId = ++state.estimateRequest;
    const summary = $("estimate");
    const start = $("start-test");
    start.disabled = true;
    if (!selectedCandidates().length || !$("suite").value) {
        summary.textContent = $("suite").value ? "请选择候选版本" : "暂无测试方案";
        return;
    }
    summary.textContent = "正在估算…";
    try {
        const data = await command("eval", "estimate", ...currentTestArgs());
        if (requestId !== state.estimateRequest) return;
        const parts = [`${data.total} 局`, `待运行 ${data.new} 局`];
        if (data.new) parts.push(duration(data.eta_seconds));
        summary.textContent = parts.join(" · ");
        start.disabled = state.starting;
    } catch (error) {
        if (requestId === state.estimateRequest) summary.textContent = error.message;
    }
}

function renderRun(run) {
    const stateLabel = statuses[run.status] || run.status;
    $("run-status").textContent = stateLabel;
    $("run-status").classList.toggle("failure", run.failed > 0);
    $("run-count").textContent = `${run.completed} / ${run.total} 局`;
    $("run-progress").max = run.total;
    $("run-progress").value = run.completed;
    const metadata = [`${run.suite.name} · 基准 ${versionLabel(run.baseline)}`];
    if (["queued", "running"].includes(run.status)) {
        metadata.push(run.eta_seconds === null ? duration(null) : `剩余 ${duration(run.eta_seconds)}`);
    }
    if (run.failed) metadata.push(`失败 ${run.failed} 局${run.timed_out ? `（超时 ${run.timed_out} 局）` : ""}`);
    $("run-meta").textContent = metadata.join(" · ");
    $("compare-delta-title").textContent = run.baseline === state.overview.main ? "比主力版" : "比基准";

    const active = ["queued", "running", "cancelling"].includes(run.status);
    const canResume = !active && run.status !== "completed" && run.pending > 0;
    const canRetry = !active && run.failed > 0;
    document.querySelector('[data-eval="resume"]').hidden = !canResume;
    document.querySelector('[data-eval="retry"]').hidden = !canRetry;
    document.querySelector('[data-eval="cancel"]').hidden = !["queued", "running"].includes(run.status);
    $("repeat-run").hidden = run.status !== "completed";
    $("publish-run").hidden = active;
    $("run-actions").hidden = false;
    selectOptions("opponent-filter", [["", "全部"], ...(run.mode === "direct" ? [run.baseline] : run.suite.opponents)
        .map((id) => [id, versionLabel(id)])]);
}

function componentsText(components) {
    return scoreItems.map(([key, label]) => `${label} ${number.format(components[key])}`).join(" · ");
}

function renderComparison(rows) {
    const body = $("comparison");
    body.replaceChildren();
    if (!rows.length) emptyTable(body, 6, "当前条件下无对比结果");
    for (const row of rows) {
        const hasScore = row.completed_pairs > 0;
        const delta = hasScore ? `${row.score_delta > 0 ? "+" : ""}${number.format(row.score_delta)}` : "—";
        addRow(body, [
            versionLabel(row.candidate), hasScore ? number.format(row.mean_score) : "—",
            element("span", delta, row.score_delta > 0 ? "positive" : row.score_delta < 0 ? "negative" : ""),
            hasScore ? percent.format(row.win_rate) : "—",
            hasScore ? element("span", componentsText(row.components), "components") : "—",
            element("span", `${row.completed_pairs} / ${row.total_pairs}`, row.missing_cases ? "failure" : ""),
        ]);
    }
}

function renderCases(data) {
    const body = $("cases");
    body.replaceChildren();
    $("case-count").textContent = data.total > data.rows.length ? `${data.rows.length} / ${data.total} 局` : `${data.total} 局`;
    if (!data.rows.length) emptyTable(body, 7, "当前条件下无对局");
    for (const row of data.rows) {
        const replay = row.has_replay ? element("a", "打开回放") : "—";
        if (replay instanceof HTMLAnchorElement) {
            replay.href = `/replay?case=${encodeURIComponent(row.case_id)}`;
            replay.target = "_blank";
            replay.rel = "noopener";
        }
        let label = row.won === undefined ? "待完成" : "已完成";
        if (row.failures.length) label = row.failures.some((failure) => failure.kind === "timeout") ? "超时" : "失败";
        addRow(body, [versionLabel(row.candidate), versionLabel(row.opponent), row.seed,
            row.side === "red" ? "红方" : "蓝方", row.score === undefined ? "—" : number.format(row.score),
            element("span", label, row.failures.length ? "failure" : ""), replay]);
    }
}

async function refreshResults() {
    clearTimeout(state.poll);
    const runId = state.runId;
    const requestId = ++state.resultRequest;
    if (!runId) return;
    const args = [];
    if ($("opponent-filter").value) args.push("--opponent", $("opponent-filter").value);
    if ($("side-filter").value) args.push("--side", $("side-filter").value);
    try {
        const [status, comparison, cases] = await Promise.all([
            command("eval", "status", runId), command("eval", "compare", runId, ...args),
            command("eval", "cases", runId, "--sort", "score_delta", ...args),
        ]);
        if (requestId !== state.resultRequest || state.page !== "results") return;
        const launching = status.status === "incomplete" && status.pending > 0 && Date.now() < state.queueDeadline;
        if (!launching) state.queueDeadline = 0;
        const run = launching ? { ...status, status: "queued" } : status;
        state.run = run;
        renderRun(run);
        renderComparison(comparison.rows);
        renderCases(cases);
        if (["queued", "running", "cancelling"].includes(run.status)) state.poll = setTimeout(refreshResults, 1200);
    } catch (error) {
        if (requestId === state.resultRequest && state.page === "results") showNotice(error.message, true);
    }
}

async function showRun(runId, queued = false) {
    state.runId = runId;
    state.run = null;
    state.queueDeadline = queued ? Date.now() + 5000 : 0;
    $("run-actions").hidden = true;
    $("run-status").textContent = "正在读取…";
    $("run-count").textContent = "";
    $("run-progress").value = 0;
    $("run-meta").textContent = "";
    $("comparison").replaceChildren();
    $("cases").replaceChildren();
    $("case-count").textContent = "";
    $("side-filter").value = "";
    selectOptions("opponent-filter", [["", "全部"]]);
    await loadOverview();
    if (location.hash === "#results") await refreshResults();
    else location.hash = "results";
}

function applyPage() {
    const target = location.hash.slice(1);
    state.page = pages.includes(target) ? target : "strategy";
    clearTimeout(state.poll);
    clearTimeout(state.trainingPoll);
    showNotice("");
    for (const id of pages) {
        $(id).hidden = id !== state.page;
        const link = document.querySelector(`nav a[href="#${id}"]`);
        if (id === state.page) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
    }
    if (!state.overview) return;
    if (state.page === "strategy") void refreshTraining();
    if (state.page === "tests") void estimate();
    if (state.page === "results") void refreshResults();
}

function showOnline(op, data) {
    const output = $("online-output");
    output.replaceChildren();
    if (op === "pack") output.append(element("p", `预检通过 · ${number.format(data.bytes)} 字节`), element("p", data.path));
    if (op === "practice") {
        output.append(element("p", `房间 ${data.room} · 已完成 ${data.matches.length} 局`));
        for (const [index, match] of data.matches.entries()) {
            const link = element("a", `第 ${index + 1} 局回放`);
            link.href = match.replay_url;
            link.target = "_blank";
            link.rel = "noopener";
            const row = element("p", "");
            row.append(link);
            output.append(row);
        }
        if (data.close_error) output.append(element("p", data.close_error, "failure"));
    }
    if (op === "submit" || op === "status") {
        $("submission-id").value = data.id;
        const evaluation = data.evaluation;
        const headline = evaluation ? `提交 ${data.id} · ${statuses[evaluation.status] || evaluation.status}` : `提交 ${data.id}`;
        output.append(element("p", headline));
        if (evaluation?.score != null) output.append(element("p", `总分 ${number.format(evaluation.score)}`));
        if (evaluation?.breakdown) output.append(element("p", componentsText(evaluation.breakdown)));
        if (evaluation?.error) output.append(element("p", evaluation.error, "failure"));
    }
    output.hidden = false;
}

$("snapshot-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void withButton(event.submitter, async () => {
        const created = await command("snapshot", "create", `--strategy=${$("source").value}`, `--note=${$("note").value}`);
        $("note").value = "";
        await loadOverview();
        showNotice(`已创建 ${versionLabel(created.id)}`);
    });
});

$("refresh-training").addEventListener("click", () => void withButton($("refresh-training"), refreshTraining));

$("test-form").addEventListener("change", (event) => {
    if (event.target === $("baseline")) renderTestSetup(false);
    void estimate();
});

$("test-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const start = $("start-test");
    if (start.disabled) return;
    state.starting = true;
    await withButton(start, async () => {
        const run = await command("eval", "start", ...currentTestArgs());
        await showRun(run.run_id, run.status === "queued");
    });
    state.starting = false;
    if (state.page === "tests") void estimate();
});

$("run-select").addEventListener("change", () => void withButton($("refresh-results"), () => showRun($("run-select").value)));
$("refresh-results").addEventListener("click", () => void withButton($("refresh-results"), async () => {
    await loadOverview();
    await refreshResults();
}));
for (const filter of ["opponent-filter", "side-filter"]) $(filter).addEventListener("change", () => void refreshResults());

for (const button of document.querySelectorAll("[data-eval]")) {
    button.addEventListener("click", () => void withButton(button, async () => {
        const run = await command("eval", button.dataset.eval, state.runId);
        if (run.status === "queued" && run.run_id === state.runId) {
            state.queueDeadline = Date.now() + 5000;
            renderRun(run);
            clearTimeout(state.poll);
            state.poll = setTimeout(refreshResults, 700);
        } else await refreshResults();
    }));
}

$("repeat-run").addEventListener("click", () => void withButton($("repeat-run"), async () => {
    const { candidates, baseline, suite, mode, shards, replay, timeout } = state.run;
    const latest = await command("suite", "show", suite.name);
    if (latest.suite_id !== suite.suite_id) throw new Error("测试方案已更新，请从测试页选择当前方案");
    const run = await command("eval", "start", ...testArgs(candidates, baseline, suite.name, mode),
        "--shards", String(shards), "--timeout", String(timeout), ...(replay ? ["--replay"] : []));
    await showRun(run.run_id, run.status === "queued");
}));

$("publish-run").addEventListener("click", () => {
    if (!window.confirm("将此批次及关联策略源码、权重共享到 Git 远端？")) return;
    void withButton($("publish-run"), async () => {
        const published = await command("sync", "publish", state.runId, "--push");
        showNotice(`已发布到 Git · ${published.completed} 局`);
    });
});

for (const button of document.querySelectorAll("[data-online]")) {
    button.addEventListener("click", () => {
        const op = button.dataset.online;
        const version = $("online-version").value;
        if (op === "submit" && !window.confirm(`将 ${versionLabel(version)} 提交官网计榜评测？`)) return;
        void withButton(button, async () => showOnline(op, await command("online", op, version)));
    });
}

$("online-status-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void withButton(event.submitter, async () => showOnline("status", await command("online", "status", $("submission-id").value.trim())));
});

window.addEventListener("hashchange", applyPage);
loadOverview().then(applyPage).catch((error) => showNotice(error.message, true));
