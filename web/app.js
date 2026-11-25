const $ = (id) => document.getElementById(id);
const pages = ["strategy", "tests", "results"];
const state = {
    page: "strategy",
    overview: null,
    runId: "",
    runSummary: null,
    runCandidate: null,
    run: null,
    poll: null,
    strategyPoll: null,
    strategyRequest: 0,
    projectsView: "",
    historyConditions: new Map(),
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
const score = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 1 });
const timestamp = new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "medium" });
const scoreItems = [["kill", "击毁"], ["damage", "伤害"], ["survive", "生存"], ["time", "时间"]];
const projectKinds = { rule: "规则", planning: "规划", rl: "强化学习", hybrid: "混合", imitation: "模仿学习", other: "其他" };
const suitesByName = { quick: "快速测试", full: "本地 40 局", acceptance: "独立验收" };
const suiteLabel = (name) => suitesByName[name] || name;

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
    if (!version) return "未载入版本";
    if (id === "baseline") return version.name;
    const project = state.overview.projects.find((item) => item.versions.some((entry) => entry.id === id));
    const entry = project?.versions.find((item) => item.id === id);
    return entry ? `${project.name} v${entry.number}` : version.name;
}

function versionText(id) {
    return element("span", versionLabel(id));
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
    return row;
}

function emptyTable(body, columns, text) {
    const cell = element("td", text, "muted");
    cell.colSpan = columns;
    const row = document.createElement("tr");
    row.append(cell);
    body.replaceChildren(row);
}

function renderTeamScore() {
    const { team_scores, sync_error, score_sync_running, team_sync } = state.overview;
    const best = team_scores?.best;
    $("team-score").textContent = best ? score.format(best.score) : "—";
    const detail = [best ? `${best.games} 局 · 四档人机` : "40 局 · 四档人机 · 暂无成绩"];
    if (score_sync_running) detail.push("同步中");
    else if (sync_error) detail.push("同步失败");
    $("team-score-meta").textContent = detail.join(" · ");
    $("team-score-meta").title = sync_error || "";
    $("team-sync-error").hidden = !team_sync?.error;
    $("team-sync-error").title = team_sync?.error || "";
}

function projectMetric(label, total, detail) {
    const metric = element("div", "", "project-metric");
    const value = element("dd", "");
    const meta = element("span", "", "score-meta");
    meta.append(...detail.map((line) => element("span", line)));
    value.append(element("strong", total === null ? "—" : score.format(total), "score-value"),
        meta);
    metric.append(element("dt", label), value);
    return metric;
}

function localResultLink(record, text, className = "") {
    const link = element("a", text, className);
    link.href = `#results?${new URLSearchParams({ run: record.run_id, candidate: record.version_id })}`;
    link.addEventListener("click", () => { state.runSummary = record; });
    return link;
}

function officialRecordButton(record) {
    const button = element("button", "官网记录", "history-link");
    button.addEventListener("click", () => {
        $("official-operations").open = true;
        showOnline("status", { id: record.id, evaluation: record });
        $("online-output").scrollIntoView({ block: "nearest" });
    });
    return button;
}

function historyVersionCell(item) {
    const label = element("div", "", "history-version");
    const version = element("strong", `v${item.number}`);
    label.append(version);
    if (item.note) label.append(element("span", item.note, "history-note"));
    const actions = element("div", "", "history-actions");
    const test = element("button", "测试此版", "history-link");
    test.addEventListener("click", () => selectForTest([item.id]));
    actions.append(test);
    if (item.id === state.overview.main) actions.append(element("span", "主力版", "tag"));
    else {
        const main = element("button", "设为主力", "history-link");
        main.addEventListener("click", () => withButton(main, async () => {
            await command("snapshot", "main", item.id);
            await loadOverview(true);
            showNotice(`已设为主力版 · ${versionLabel(item.id)}`);
        }));
        actions.append(main);
    }
    label.append(actions);
    return label;
}

function historyOfficialCell(official, bestScore) {
    const { best, latest } = official;
    const cell = element("div", "", "history-score");
    if (best) {
        cell.append(element("strong", score.format(best.score)),
            element("span", `${best.games} 局${best.score === bestScore ? " · 最高" : ""}`, "muted"),
            officialRecordButton(best));
    } else if (latest) {
        cell.append(element("span", statuses[latest.status] || latest.status, "muted"), officialRecordButton(latest));
    } else cell.append(element("span", "暂无官网成绩", "muted"));
    return cell;
}

function completeResult(record) {
    return record?.status === "completed" && record.scored_games === record.scheduled_games;
}

function historyLocalCell(record, selected, highest) {
    const cell = element("div", "", "history-score");
    if (!record) {
        cell.append(element("span", selected ? "此条件暂无成绩" : "暂无本地成绩", "muted"));
        return cell;
    }
    const detail = `${record.scored_games} / ${record.scheduled_games} 局`;
    cell.append(element("strong", record.total_score === null ? "—" : score.format(record.total_score)),
        element("span", completeResult(record) ? detail : `${detail} · 未完成`, "muted"));
    if (completeResult(record)) {
        const delta = record.total_score - highest;
        cell.append(delta === 0 ? element("span", "最高", "tag") : element("span", `比最高 ${score.format(delta)}`, "muted"));
    }
    cell.append(localResultLink(record, "本地结果", "history-link"));
    return cell;
}

function conditionOptions(conditions) {
    const descriptions = conditions.map((record) => {
        const baseline = versionLabel(record.baseline);
        const opponent = record.opponents.map((id) => versionLabel(id)).join("、");
        const match = record.mode === "direct" ? `交手 ${baseline}` : `陪练 ${opponent} · 基准 ${baseline}`;
        return `${suiteLabel(record.suite)} · ${record.scheduled_games} 局 · ${match}`;
    });
    const detailed = descriptions.map((label, index) => {
        const condition = conditions[index];
        if (descriptions.indexOf(label) === descriptions.lastIndexOf(label)) return label;
        return `${label} · ${condition.rules_version} · 限时 ${condition.timeout} 秒`;
    });
    return detailed.map((label, index) => {
        const id = conditions[index].condition_id;
        return new Option(detailed.indexOf(label) === detailed.lastIndexOf(label) ? label : `${label} · ${id.slice(-8)}`, id);
    });
}

function historyContents(project) {
    const content = element("div", "");
    const conditions = new Map();
    for (const item of project.versions) {
        for (const record of item.local.by_condition) conditions.set(record.condition_id, record);
    }
    const select = document.createElement("select");
    if (conditions.size) {
        const saved = state.historyConditions.get(project.id);
        const recent = project.local.best_by_condition[0] || project.local.latest;
        select.append(...conditionOptions([...conditions.values()]));
        select.value = conditions.has(saved) ? saved : recent.condition_id;
        const label = element("label", "本地测试条件", "history-filter");
        label.append(select);
        content.append(label);
    }

    const scroller = element("div", "", "table-scroll");
    const table = element("table", "", "history-table");
    const headers = table.createTHead().insertRow();
    for (const title of ["版本", "官网最高总分", "本地实测总分"]) {
        const header = element("th", title);
        header.scope = "col";
        headers.append(header);
    }
    const body = table.createTBody();
    function render() {
        const selected = select.value;
        state.historyConditions.set(project.id, selected);
        const rows = newestFirst(project).map((item) => {
            const record = item.local.by_condition.find((value) => value.condition_id === selected);
            return { item, record };
        });
        const completed = rows.filter(({ record }) => completeResult(record));
        const highest = completed.length ? Math.max(...completed.map(({ record }) => record.total_score)) : null;
        body.replaceChildren();
        for (const { item, record } of rows) addRow(body, [historyVersionCell(item),
            historyOfficialCell(item.official, project.official.best?.score), historyLocalCell(record, selected, highest)]);
    }
    select.addEventListener("change", render);
    render();
    scroller.append(table);
    content.append(scroller);
    return content;
}

function versionHistory(project, expanded) {
    const history = element("details", "", "project-history");
    history.dataset.projectId = project.id;
    history.open = expanded;
    history.append(element("summary", `历史版本 · ${project.versions.length}`, "utility-toggle"));
    let content = null;
    const sync = () => {
        if (history.open && !content) {
            content = historyContents(project);
            history.append(content);
        } else if (!history.open && content) {
            content.remove();
            content = null;
        }
    };
    history.addEventListener("toggle", sync);
    if (expanded) sync();
    return history;
}

function renderProjects() {
    const projects = state.overview.projects;
    const view = JSON.stringify([projects, state.overview.main]);
    if (view === state.projectsView) return;
    state.projectsView = view;
    const container = $("projects");
    const expanded = new Set([...container.querySelectorAll("details[open]")].map((item) => item.dataset.projectId));
    container.replaceChildren();
    if (!projects.length) container.append(element("p", "暂无策略项目", "panel empty"));
    for (const project of projects) {
        const card = element("article", "", "panel project-card");
        const header = element("div", "", "project-heading");
        header.append(element("h2", project.name), element("span", projectKinds[project.kind] || project.kind, "project-kind"));
        const metrics = element("dl", "", "project-metrics");
        const best = project.official.best;
        const latest = project.local.latest;
        const officialDetail = [best ? `${best.games} 局 · 四档人机` : "暂无成绩"];
        const localDetail = latest ? [
            `${suiteLabel(latest.suite)} · ${latest.scored_games} / ${latest.scheduled_games} 局${latest.status === "completed" ? "" : " · 未完成"}`,
            `${latest.mode === "direct" ? "对手" : "陪练"} ${latest.opponents.map((id) => versionLabel(id)).join("、")}`,
        ] : ["暂无测试"];
        if (completeResult(latest)) {
            const comparable = project.local.best_by_condition.find((record) => record.condition_id === latest.condition_id);
            if (comparable && comparable.total_score > latest.total_score) {
                localDetail.push(`比同条件最高低 ${score.format(comparable.total_score - latest.total_score)} 分`);
            }
        }
        metrics.append(projectMetric("官网最高总分", best ? best.score : null, officialDetail),
            projectMetric("最近本地实测总分", latest?.total_score ?? null, localDetail));
        card.append(header, metrics);

        const current = project.training.latest?.snapshot_id || latest?.version_id || project.versions[0]?.id;
        const actions = element("div", "", "project-actions");
        if (project.sources.length || current) {
            const test = element("button", project.sources.length ? "测试当前代码" : "测试项目版本", "primary");
            test.addEventListener("click", () => withButton(test, async () => {
                if (!project.sources.length) return selectForTest([current]);
                const versions = await Promise.all(project.sources.map((source) => command("snapshot", "create", `--strategy=${source}`)));
                await loadOverview();
                selectForTest(versions.map(({ id }) => id));
            }));
            actions.append(test);
        }
        if (latest) actions.append(localResultLink(latest, "查看本地结果", "button-link"));
        if (actions.childElementCount) card.append(actions);
        if (project.versions.length) card.append(versionHistory(project, expanded.has(project.id)));
        container.append(card);
    }
}

function newestFirst(project) {
    return [...project.versions].sort((a, b) => b.number - a.number);
}

function versionGroups(includeBot) {
    const groups = state.overview.projects.filter((p) => p.versions.length);
    if (!includeBot) return groups;
    const bot = state.overview.versions.find((v) => v.id === "baseline");
    return [{ id: "", name: bot.name, versions: [bot] }, ...groups];
}

function shortVersion(v) {
    return v.id === "baseline" ? v.name : `v${v.number}${v.note ? " · " + v.note : ""}`;
}

function versionPicker(projectId, versionId, groups, preferred) {
    const current = $(projectId).value;
    const preferredGroup = preferred && groups.find((p) => p.versions.some((v) => v.id === preferred));
    selectOptions(projectId, groups.map((p) => [p.id, p.name]), preferredGroup ? preferredGroup.id : current);
    const group = groups.find((p) => p.id === $(projectId).value);
    const entries = group ? newestFirst(group).map((v) => [v.id, shortVersion(v)]) : [];
    selectOptions(versionId, entries, preferred || $(versionId).value);
    $(versionId).closest("label").hidden = group?.id === "";
}

function renderOnlineVersions() {
    versionPicker("online-project", "online-version", versionGroups(false));
    for (const button of document.querySelectorAll("[data-online]")) button.disabled = !$("online-version").value;
}

function selectedCandidates() {
    return [...new Set([...$("candidates").querySelectorAll("input:checked")].map((chosen) => chosen.value))];
}

function markCandidateCount(group) {
    const count = group.querySelectorAll("input:checked").length;
    group.querySelector("summary").textContent = group.dataset.name + (count ? ` · 已选 ${count}` : "");
}

function renderTestSetup(forceMain) {
    const groups = versionGroups(true);
    versionPicker("baseline-project", "baseline", groups, forceMain ? state.overview.main : null);
    selectOptions("suite", state.overview.suites.map((suite) => [suite, suiteLabel(suite)]), $("suite").value || "quick");
    const selected = new Set(selectedCandidates());
    const choices = $("candidates");
    const opened = new Set([...choices.querySelectorAll("details[open]")].map((g) => g.dataset.project));
    choices.replaceChildren();
    for (const group of groups) {
        const entries = newestFirst(group).filter((v) => v.id !== $("baseline").value);
        if (!entries.length) continue;
        const section = element("details", "", "candidate-project");
        section.dataset.project = group.id;
        section.dataset.name = group.name;
        section.open = opened.has(group.id);
        section.append(element("summary", group.name));
        for (const v of entries) {
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.value = v.id;
            checkbox.checked = selected.has(v.id);
            const choice = element("label", "", "check");
            choice.append(checkbox, shortVersion(v));
            section.append(choice);
        }
        markCandidateCount(section);
        choices.append(section);
    }
    if (!choices.childElementCount) choices.append(element("p", "暂无可选版本"));
}

async function loadOverview(forceMain = false, page = null) {
    const overview = await request("/api/overview");
    if (page && state.page !== page) return;
    state.overview = overview;
    renderTeamScore();
    renderProjects();
    renderTraining();
    renderOnlineVersions();
    renderTestSetup(forceMain);
    const runs = [...state.overview.runs];
    if (state.runSummary && !runs.some(({ run_id }) => run_id === state.runSummary.run_id)) runs.push(state.runSummary);
    if (!runs.some(({ run_id }) => run_id === state.runId)) state.runId = runs[0]?.run_id || "";
    selectOptions("run-select", runs.map((run) => [
        run.run_id, run.created_at ? `${suiteLabel(run.suite)} · ${timestamp.format(new Date(run.created_at * 1000))}` : "链接中的批次",
    ]), state.runId);
    $("refresh-results").disabled = !state.runId;
    $("no-runs").hidden = Boolean(state.runId);
    $("run-content").hidden = !state.runId;
}

function selectForTest(versionIds) {
    const baseline = $("baseline");
    if (versionIds.includes(baseline.value)) {
        $("baseline-project").value = "";
        renderTestSetup(false);
    }
    for (const candidate of $("candidates").querySelectorAll("input")) candidate.checked = versionIds.includes(candidate.value);
    for (const group of $("candidates").querySelectorAll("details")) {
        group.open = Boolean(group.querySelector("input:checked"));
        markCandidateCount(group);
    }
    location.hash = "tests";
}

function renderTraining() {
    const { projects, versions } = state.overview;
    const jobs = projects.filter(({ training }) => training.latest).map(({ name, training }) => ({
        ...training.latest, project: name,
    }));
    const active = projects.reduce((total, { training }) => total + training.active, 0);
    const view = JSON.stringify([jobs, active, versions.map(({ id }) => id)]);
    if (view === state.trainingView) return;
    state.trainingView = view;
    $("training-panel").hidden = jobs.length === 0;
    $("training-summary").textContent = active ? `${active} 项进行中` : `${jobs.length} 个项目`;
    const body = $("training-jobs");
    body.replaceChildren();
    for (const job of jobs) {
        const status = element("span", trainingStatuses[job.status] || job.status);
        if (job.error) status.append(element("span", job.error, "training-error"));
        let output = job.checkpoint_id ? "已登记检查点" : "—";
        if (job.snapshot_id && versions.some(({ id }) => id === job.snapshot_id)) {
            const button = element("button", "测试此版本");
            button.addEventListener("click", () => selectForTest([job.snapshot_id]));
            output = element("div", versionLabel(job.snapshot_id), "training-output");
            output.append(button);
        }
        addRow(body, [job.project, timestamp.format(new Date(job.created_at * 1000)), status, output]);
    }
}

async function refreshStrategy(fresh = false) {
    clearTimeout(state.strategyPoll);
    const request = ++state.strategyRequest;
    try {
        if (!fresh) await loadOverview(false, "strategy");
    } catch (error) {
        if (state.page === "strategy" && request === state.strategyRequest) showNotice(error.message, true);
    } finally {
        if (state.page === "strategy" && request === state.strategyRequest) {
            state.strategyPoll = setTimeout(refreshStrategy, 15000);
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
    const direct = $("mode").value === "direct";
    $("baseline-label").textContent = direct ? "指定对手" : "参照项目";
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
        const selected = versionLabel($("baseline").value);
        const matchup = direct ? `各候选分别对战 ${selected}` : `各候选与 ${selected} 分别打方案中的陪练`;
        const parts = [matchup, `共 ${data.total} 局`];
        if (data.total > data.new) parts.push(`复用 ${data.total - data.new} 局`);
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
    const direct = run.mode === "direct";
    const base = versionLabel(run.baseline);
    const opponents = run.suite.opponents.map((id) => versionLabel(id)).join("、");
    const matchup = direct ? `各候选分别对战 ${base}` : `各候选与 ${base} 分别对战 ${opponents}`;
    $("compare-title").textContent = direct ? `对战成绩 · 对手：${base}` : `陪练成绩 · 参照：${base}`;
    $("delta-heading").textContent = direct ? "与对手分差" : "与参照版分差";
    const metadata = [suiteLabel(run.suite.name), matchup];
    if (["queued", "running"].includes(run.status)) {
        metadata.push(run.eta_seconds === null ? duration(null) : `剩余 ${duration(run.eta_seconds)}`);
    }
    if (run.failed) metadata.push(`失败 ${run.failed} 局${run.timed_out ? `（超时 ${run.timed_out} 局）` : ""}`);
    $("run-meta").textContent = metadata.join(" · ");
    $("cases-title").textContent = state.runCandidate ? `对局明细 · ${versionLabel(state.runCandidate)}` : "对局明细";
    if (state.runSummary?.run_id === run.run_id && !state.runSummary.created_at) {
        $("run-select").selectedOptions[0].textContent = suiteLabel(run.suite.name);
    }
    const active = ["queued", "running", "cancelling"].includes(run.status);
    const canResume = !active && run.status !== "completed" && run.pending > 0;
    const canRetry = !active && run.failed > 0;
    document.querySelector('[data-eval="resume"]').hidden = !canResume;
    document.querySelector('[data-eval="retry"]').hidden = !canRetry;
    document.querySelector('[data-eval="cancel"]').hidden = !["queued", "running"].includes(run.status);
    $("repeat-run").hidden = run.status !== "completed";
    $("run-actions").hidden = !(canResume || canRetry || run.status === "completed" || ["queued", "running"].includes(run.status));
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
    const order = state.run.candidates;
    for (const row of [...rows].sort((left, right) => order.indexOf(left.candidate) - order.indexOf(right.candidate))) {
        const delta = row.total_delta;
        const deltaLabel = delta == null ? "—" : `${delta > 0 ? "+" : ""}${score.format(delta)}`;
        const played = `${row.scored_games} / ${row.scheduled_games} 局`;
        const entry = addRow(body, [
            versionText(row.candidate), element("strong", row.scored_games ? score.format(row.total_score) : "—", "result-score"),
            element("span", played, row.scored_games < row.scheduled_games ? "muted" : ""),
            element("span", deltaLabel, delta > 0 ? "positive" : delta < 0 ? "negative" : ""),
            row.completed_pairs ? percent.format(row.win_rate) : "—",
            element("span", `${row.completed_pairs} / ${row.total_pairs}`, row.missing_cases ? "failure" : ""),
        ]);
        entry.classList.toggle("focused-result", row.candidate === state.runCandidate);
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
        const completed = row.score !== undefined;
        const failure = completed ? null : row.failures.find((item) => item.case_id === row.case_id);
        let label = completed ? "已完成" : "待完成";
        if (failure) label = failure.kind === "timeout" ? "超时" : "失败";
        addRow(body, [versionText(row.candidate), versionText(row.opponent), row.seed,
            row.side === "red" ? "红方" : "蓝方", completed ? number.format(row.score) : "—",
            element("span", label, failure ? "failure" : ""), replay]);
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
    const candidate = state.runCandidate ? ["--candidate", state.runCandidate] : [];
    try {
        const [status, comparison, cases] = await Promise.all([
            command("eval", "status", runId), command("eval", "compare", runId, ...args),
            command("eval", "cases", runId, "--sort", "score_delta", ...args, ...candidate),
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

async function showRun(runId, queued = false, summary = null, route = null) {
    state.runId = runId;
    state.runSummary = summary;
    state.runCandidate = route?.candidate || null;
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
    if (location.hash === "#results" || route) await refreshResults();
    else location.hash = "results";
}

function applyPage(fresh = false) {
    const [target, query] = location.hash.slice(1).split("?", 2);
    state.page = pages.includes(target) ? target : "strategy";
    clearTimeout(state.poll);
    clearTimeout(state.strategyPoll);
    showNotice("");
    for (const id of pages) {
        $(id).hidden = id !== state.page;
        const link = document.querySelector(`nav a[href="#${id}"]`);
        if (id === state.page) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
    }
    if (!state.overview) return;
    if (state.page === "strategy") void refreshStrategy(fresh);
    if (state.page === "tests") void estimate();
    if (state.page === "results") {
        const params = new URLSearchParams(query);
        const linkedRun = params.get("run");
        if (linkedRun) {
            const summary = state.runSummary?.run_id === linkedRun ? state.runSummary : { run_id: linkedRun };
            window.scrollTo(0, 0);
            showRun(linkedRun, false, summary, { candidate: params.get("candidate") }).catch((error) => showNotice(error.message, true));
        } else {
            state.runCandidate = null;
            void refreshResults();
        }
    }
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
            const row = element("p", `我方 ${score.format(match.score)} 分 · `);
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
        if (evaluation?.score != null) output.append(element("p", `总分 ${score.format(evaluation.score)}`));
        if (evaluation?.breakdown) output.append(element("p", componentsText(evaluation.breakdown)));
        if (evaluation?.error) output.append(element("p", evaluation.error, "failure"));
    }
    output.hidden = false;
}

$("online-project").addEventListener("change", renderOnlineVersions);

$("test-form").addEventListener("change", (event) => {
    if (event.target === $("baseline") || event.target === $("baseline-project")) renderTestSetup(false);
    if (event.target.closest("#candidates")) {
        for (const box of $("candidates").querySelectorAll("input")) {
            if (box.value === event.target.value) box.checked = event.target.checked;
        }
        for (const group of $("candidates").querySelectorAll("details")) markCandidateCount(group);
    }
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

for (const button of document.querySelectorAll("[data-online]")) {
    button.addEventListener("click", () => {
        const op = button.dataset.online;
        const version = $("online-version").value;
        if (op === "submit" && !window.confirm(`将 ${versionLabel(version)} 提交官网计榜评测？`)) return;
        const args = op === "practice" ? ["--level", $("practice-level").value] : [];
        void withButton(button, async () => showOnline(op, await command("online", op, version, ...args)));
    });
}

$("online-status-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void withButton(event.submitter, async () => showOnline("status", await command("online", "status", $("submission-id").value.trim())));
});

window.addEventListener("hashchange", () => applyPage());
loadOverview().then(() => applyPage(true)).catch((error) => showNotice(error.message, true));
