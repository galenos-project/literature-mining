(function () {
  "use strict";

  const PLOTLY_FONT = { family: "Inter, sans-serif", color: "#1B2233", size: 12 };
  const LINE_PALETTE = ["#2B6E68", "#D98A3D", "#B5423A", "#4C6B8A", "#8A6D3B", "#5C8A6B", "#946E9A", "#3D5A80"];

  function fillUrl(template, params) {
    let out = template;
    for (const [key, val] of Object.entries(params)) {
      out = out.replace(`__${key}__`, val);
    }
    return out;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  // Escape `text`, then wrap the first case-insensitive occurrence of
  // `query` in <mark>. Slices the raw string first so the <mark> tags are
  // never themselves escaped.
  function highlightMatch(text, query) {
    const i = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
    if (i < 0) return escapeHtml(text);
    return (
      escapeHtml(text.slice(0, i)) +
      "<mark>" + escapeHtml(text.slice(i, i + query.length)) + "</mark>" +
      escapeHtml(text.slice(i + query.length))
    );
  }

  function matchingKeywords(topic, query, max) {
    const q = query.toLowerCase();
    return (topic.keywords || [])
      .filter((k) => k.toLowerCase().includes(q))
      .slice(0, max || 6);
  }

  function monthToParts(monthStr) {
    // monthStr like '2023-04-01'
    const [y, m] = monthStr.split("-");
    return { year: parseInt(y, 10), month: parseInt(m, 10) };
  }

  function goToMonth(urls, topicId, monthStr) {
    const { year, month } = monthToParts(monthStr);
    window.location.href = fillUrl(urls.monthPage, { ID: topicId, YEAR: year, MONTH: month });
  }

  function baseLayout(overrides) {
    return Object.assign(
      {
        font: PLOTLY_FONT,
        margin: { t: 10, r: 20, b: 40, l: 50 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        legend: { orientation: "h", y: -0.2 },
        xaxis: { gridcolor: "#E8EAED", zeroline: false },
        yaxis: { gridcolor: "#E8EAED", zeroline: false, rangemode: "tozero" },
        hovermode: "closest",
      },
      overrides || {}
    );
  }

  const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

  // ---- Study-type breakdown (index + topic pages) -------------------------

  // Validated pair (dataviz validate_palette.js, light, on #FFFFFF): published
  // takes a higher-chroma step of the site teal, grey literature a violet --
  // not amber, which means "predicted" elsewhere on the site.
  const STUDY_TYPE_SERIES = [
    { key: "published", name: "Published", color: "#00897B" },
    { key: "grey", name: "Grey literature", color: "#7B5FC4" },
    { key: "unrecorded", name: "Venue not recorded", color: "#B7BBC2" },
  ];

  // Horizontal bars, one per study type grouped by family, each split into
  // published / grey literature, with the type's share of all papers at the
  // bar tip. `rows` comes from db.get_study_type_counts.
  function renderStudyTypeChart(elId, rows) {
    const el = document.getElementById(elId);
    if (!el || !rows || !rows.length) return;
    const grandTotal = rows.reduce((sum, r) => sum + r.total, 0);
    const pct = (n, of) => (of ? (100 * n / of).toFixed(of >= 1000 ? 1 : 0) : "0") + "%";
    const y = [rows.map((r) => r.family), rows.map((r) => r.label)];

    const traces = STUDY_TYPE_SERIES
      .filter((s) => s.key !== "unrecorded" || rows.some((r) => r.unrecorded > 0))
      .map((s) => ({
        type: "bar",
        orientation: "h",
        name: s.name,
        x: rows.map((r) => r[s.key]),
        y,
        marker: { color: s.color, line: { color: "#FFFFFF", width: 2 } },
        customdata: rows.map((r) => [r.label, pct(r[s.key], r.total), r.total.toLocaleString()]),
        hovertemplate:
          `<b>%{customdata[0]}</b><br>${s.name}: %{x:,} (%{customdata[1]} of this type)` +
          "<br>All: %{customdata[2]}<extra></extra>",
      }));

    const maxTotal = Math.max(...rows.map((r) => r.total));
    const annotations = rows.map((r) => ({
      x: r.total,
      y: [r.family, r.label],
      text: pct(r.total, grandTotal),
      xanchor: "left",
      xshift: 6,
      showarrow: false,
      font: { family: "Inter, sans-serif", size: 12, color: "#3D4457" },
    }));

    Plotly.newPlot(
      elId,
      traces,
      baseLayout({
        barmode: "stack",
        bargap: 0.38,
        barcornerradius: 4,
        height: 120 + rows.length * 34,
        margin: { t: 30, r: 60, b: 40, l: 10 },
        legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", traceorder: "normal" },
        xaxis: {
          title: "Papers",
          gridcolor: "#E8EAED",
          zeroline: false,
          rangemode: "tozero",
          range: [0, maxTotal * 1.12],
          tickformat: maxTotal >= 10000 ? "~s" : ",d",
        },
        yaxis: { autorange: "reversed", automargin: true, tickfont: { color: "#1B2233" } },
        annotations,
      }),
      PLOTLY_CONFIG
    );
  }

  // ---- Topic evidence map (index page) ------------------------------------

  const EVIDENCE_COLORS = { trending: "#B5423A", established: "#B7BBC2" };
  const EVIDENCE_LABELS_PER_AXIS = 3;

  function truncate(s, n) {
    return s.length > n ? s.slice(0, n - 1).trimEnd() + "\u2026" : s;
  }

  // One bubble per topic: x = share of trials + systematic reviews, y = share
  // of pre-clinical (animal + in vitro) studies, area = papers. Established
  // topics sit behind in grey; the trending topics furthest along each axis
  // are labelled. `topics` comes from db.get_topic_evidence.
  function renderEvidenceMap(elId, topics) {
    const el = document.getElementById(elId);
    if (!el || !topics || !topics.length) return;
    const pts = topics.map((t) => {
      const c = t.counts, n = t.n_classified;
      const share = (k) => c[k] / n;
      return {
        t,
        x: share("interventional") + share("systematic_review"),
        y: share("animal") + share("in_vitro"),
        detail: [
          ["Trials", share("interventional")],
          ["Systematic reviews", share("systematic_review")],
          ["Observational", share("observational")],
          ["Qualitative", share("qualitative")],
          ["Case reports", share("case_report")],
          ["Pre-clinical", share("animal") + share("in_vitro")],
          ["Other reviews", share("other_review")],
          ["Protocols / other", share("protocol") + share("other")],
        ],
      };
    });
    const maxN = Math.max(...topics.map((t) => t.n_classified));
    const MAX_DIAMETER = 34;
    const sizeref = (2 * maxN) / (MAX_DIAMETER * MAX_DIAMETER);
    const pctf = (v) => (100 * v).toFixed(0) + "%";

    const trace = (group, isTrendy) => ({
      type: "scatter",
      mode: "markers",
      name: isTrendy ? "Trending topic" : "Other topic",
      x: group.map((p) => p.x),
      y: group.map((p) => p.y),
      customdata: group.map((p) => p.t.topic_id),
      text: group.map((p) =>
        `<b>${escapeHtml(p.t.name)}</b><br>${p.t.n_classified.toLocaleString()} papers` +
        (isTrendy ? " \u00b7 trending" : "") + "<br>" +
        p.detail.map(([k, v]) => `${k}: ${pctf(v)}`).join("<br>")
      ),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: group.map((p) => p.t.n_classified),
        sizemode: "area",
        sizeref,
        sizemin: 4,
        color: isTrendy ? EVIDENCE_COLORS.trending : EVIDENCE_COLORS.established,
        opacity: isTrendy ? 0.78 : 0.5,
        line: { color: "#FFFFFF", width: isTrendy ? 1.5 : 1 },
      },
    });
    const trending = pts.filter((p) => p.t.is_trendy);
    const established = pts.filter((p) => !p.t.is_trendy);

    // Label the trending topics furthest right and furthest up.
    // Those furthest right sit near the bottom edge, so their labels stack
    // upwards to the left; those furthest up stack downwards to the right.
    const pick = (key) => [...trending].sort((a, b) => b[key] - a[key]).slice(0, EVIDENCE_LABELS_PER_AXIS);
    const byX = pick("x");
    const byY = pick("y").filter((p) => !byX.includes(p));
    const xMax = Math.min(1, Math.max(...pts.map((p) => p.x)) * 1.08 + 0.02);
    const yMax = Math.min(1, Math.max(...pts.map((p) => p.y)) * 1.05 + 0.03);
    const label = (p, ax, ay, xanchor) => ({
      x: p.x, y: p.y, text: escapeHtml(truncate(p.t.name, 48)),
      showarrow: true, arrowhead: 0, arrowwidth: 1, arrowcolor: "#6B7280",
      ax, ay, xanchor, yanchor: "middle",
      font: { family: "Inter, sans-serif", size: 11, color: "#1B2233" },
      bgcolor: "rgba(255,255,255,0.9)",
    });
    const annotations = [
      ...[...byX].sort((a, b) => a.x - b.x).map((p, i) => label(p, -30, -45 - i * 26, "right")),
      ...[...byY].sort((a, b) => b.y - a.y).map((p, i) => label(p, 40, -6 + i * 26, "left")),
    ];

    Plotly.newPlot(
      elId,
      [trace(established, false), trace(trending, true)],
      baseLayout({
        height: 560,
        margin: { t: 30, r: 20, b: 55, l: 65 },
        legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", traceorder: "reversed", itemsizing: "constant" },
        xaxis: {
          title: "Trials and systematic reviews (share of the topic's papers)",
          tickformat: ".0%", range: [-0.01, xMax], gridcolor: "#E8EAED", zeroline: false,
        },
        yaxis: {
          title: "Pre-clinical studies (share)",
          tickformat: ".0%", range: [-0.02, yMax], gridcolor: "#E8EAED", zeroline: false,
        },
        annotations,
      }),
      PLOTLY_CONFIG
    );
    el.on("plotly_click", (data) => {
      const id = data.points[0].customdata;
      window.location.href = fillUrl(window.GALENOS.urls.topicPage, { ID: id });
    });
  }

  // Click-to-sort for tables marked data-sortable: header cells carry
  // data-sort="text" | "num"; numeric cells may carry data-value.
  function initSortableTables() {
    document.querySelectorAll("table[data-sortable]").forEach((table) => {
      const headers = [...table.tHead.rows[0].cells];
      headers.forEach((th, col) => {
        if (!th.dataset.sort) return;
        th.classList.add("sortable");
        th.tabIndex = 0;
        const sort = () => {
          const numeric = th.dataset.sort === "num";
          // First click: numbers high-to-low, text A-Z; further clicks toggle.
          const current = th.getAttribute("aria-sort");
          const desc = current ? current === "ascending" : numeric;
          headers.forEach((h) => h.removeAttribute("aria-sort"));
          th.setAttribute("aria-sort", desc ? "descending" : "ascending");
          const body = table.tBodies[0];
          const key = (row) => {
            const cell = row.cells[col];
            return numeric ? parseFloat(cell.dataset.value ?? cell.textContent) : cell.textContent.trim().toLowerCase();
          };
          const rows = [...body.rows].sort((a, b) => {
            const ka = key(a), kb = key(b);
            const cmp = ka < kb ? -1 : ka > kb ? 1 : 0;
            return desc ? -cmp : cmp;
          });
          rows.forEach((r) => body.appendChild(r));
        };
        th.addEventListener("click", sort);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } });
      });
    });
  }

  // ---- Index page --------------------------------------------------------

  function initIndexPage() {
    const urls = window.GALENOS.urls;
    let allTopics = [];
    let topicsById = new Map();
    let selectedTopicIds = []; // ordered, de-duplicated topic ids currently shown below
    let trendChartTopicIds = []; // curveNumber -> topic_id, for click-through
    let trendSortMode = "rank";
    let trendTopicCount = 10;

    fetch(urls.topics)
      .then((r) => r.json())
      .then((topics) => {
        allTopics = topics;
        topicsById = new Map(topics.map((t) => [Number(t.topic_id), t]));
        populateTopicSelect(topics);
        initTopicSearch();
        renderTrendChart();
        // Default to just the single top-trending topic (defaultTopicIds
        // is already ordered by trend_rank desc, same list the trend
        // chart's default view uses) rather than the whole top-N.
        const topTrending = (window.GALENOS.defaultTopicIds || [])[0];
        const fallback = topics[0] && topics[0].topic_id;
        const initialId = topTrending != null ? Number(topTrending) : fallback != null ? Number(fallback) : null;
        selectedTopicIds = initialId != null && topicsById.has(initialId) ? [initialId] : [];
        renderSelectedTopics();
      });

    document.getElementById("topic-select").addEventListener("change", (e) => {
      if (e.target.value) addToSelection(e.target.value);
      e.target.value = ""; // reset to the placeholder so it can be used again
      renderSelectedTopics();
    });

    document.getElementById("add-trendy-btn").addEventListener("click", () => {
      allTopics.filter((t) => t.is_trendy).forEach((t) => addToSelection(t.topic_id));
      renderSelectedTopics();
    });

    document.getElementById("clear-selected-btn").addEventListener("click", () => {
      selectedTopicIds = [];
      renderSelectedTopics();
    });

    document.querySelectorAll(".sort-toggle__btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.classList.contains("is-active")) return;
        document.querySelectorAll(".sort-toggle__btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        trendSortMode = btn.dataset.sort;
        renderTrendChart();
      });
    });

    document.getElementById("trend-count-select").addEventListener("change", (e) => {
      trendTopicCount = parseInt(e.target.value, 10);
      renderTrendChart();
    });

    function isSelected(id) {
      return selectedTopicIds.includes(Number(id));
    }

    function addToSelection(id) {
      id = Number(id);
      if (topicsById.has(id) && !selectedTopicIds.includes(id)) selectedTopicIds.push(id);
    }

    function toggleSelection(id) {
      id = Number(id);
      const idx = selectedTopicIds.indexOf(id);
      if (idx >= 0) selectedTopicIds.splice(idx, 1);
      else if (topicsById.has(id)) selectedTopicIds.push(id);
      renderSelectedTopics();
    }

    function populateTopicSelect(topics) {
      const select = document.getElementById("topic-select");
      select.innerHTML = '<option value="" selected disabled>Choose a topic\u2026</option>';
      topics.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.topic_id;
        opt.textContent = (t.is_trendy ? "\u25B2 " : "") + t.name;
        select.appendChild(opt);
      });
    }

    // Free-text topic search: filters the already-loaded `allTopics` by
    // name OR keyword and shows a dropdown of matches. Clicking a result
    // (or its checkbox) toggles it into the multi-topic selection below;
    // the dropdown stays open so several topics can be added in a row.
    function initTopicSearch() {
      const input = document.getElementById("topic-search");
      const results = document.getElementById("topic-search-results");
      if (!input || !results) return;
      let matches = [];
      let activeIndex = -1;

      function query() {
        return input.value.trim();
      }

      function search(q) {
        const needle = q.toLowerCase();
        return allTopics
          .filter(
            (t) =>
              (t.name || "").toLowerCase().includes(needle) ||
              (t.keywords || []).some((k) => k.toLowerCase().includes(needle))
          )
          .slice(0, 25);
      }

      function close() {
        results.hidden = true;
        input.setAttribute("aria-expanded", "false");
        activeIndex = -1;
      }

      function render() {
        const q = query();
        results.innerHTML = "";
        activeIndex = -1;
        if (!q) return close();

        if (!matches.length) {
          const li = document.createElement("li");
          li.className = "topic-search__empty";
          li.textContent = "No topics match \u201C" + q + "\u201D";
          results.appendChild(li);
        } else {
          const addAllLi = document.createElement("li");
          const addAllBtn = document.createElement("button");
          addAllBtn.type = "button";
          addAllBtn.className = "topic-search__addall";
          addAllBtn.textContent =
            "+ Add all " + matches.length + " result" + (matches.length === 1 ? "" : "s");
          addAllBtn.addEventListener("click", () => {
            matches.forEach((t) => addToSelection(t.topic_id));
            renderSelectedTopics();
            render(); // refresh checkmarks without closing the dropdown
          });
          addAllLi.appendChild(addAllBtn);
          results.appendChild(addAllLi);

          matches.forEach((t) => {
            const li = document.createElement("li");
            li.setAttribute("role", "option");
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "topic-search__result";
            btn.setAttribute("aria-selected", String(isSelected(t.topic_id)));
            const kws = matchingKeywords(t, q);
            btn.innerHTML =
              '<span class="topic-search__result-check" aria-hidden="true">' +
              (isSelected(t.topic_id) ? "\u2611" : "\u2610") +
              "</span>" +
              (t.is_trendy ? "\u25B2 " : "") +
              highlightMatch(t.name || "(unnamed topic)", q) +
              (kws.length
                ? '<span class="topic-search__result-kw">' +
                  kws.map((k) => highlightMatch(k, q)).join(", ") +
                  "</span>"
                : "");
            btn.addEventListener("click", () => {
              toggleSelection(t.topic_id);
              render(); // stays open, checkmark flips, so more can be added
            });
            li.appendChild(btn);
            results.appendChild(li);
          });
        }
        results.hidden = false;
        input.setAttribute("aria-expanded", "true");
      }

      function moveActive(delta) {
        const items = [...results.querySelectorAll(".topic-search__result")];
        if (!items.length) return;
        activeIndex = (activeIndex + delta + items.length) % items.length;
        items.forEach((it, i) => it.classList.toggle("is-active", i === activeIndex));
        items[activeIndex].scrollIntoView({ block: "nearest" });
      }

      input.addEventListener("input", () => {
        matches = query() ? search(query()) : [];
        render();
      });

      input.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          moveActive(1);
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          moveActive(-1);
        } else if (e.key === "Enter") {
          if (matches.length) {
            e.preventDefault();
            toggleSelection(matches[activeIndex >= 0 ? activeIndex : 0].topic_id);
            render();
          }
        } else if (e.key === "Escape") {
          input.value = "";
          matches = [];
          close();
        }
      });

      document.addEventListener("click", (e) => {
        if (!e.target.closest(".topic-search")) close();
      });
    }

    function topicIdsForTrendChart() {
      const count = trendTopicCount;
      if (trendSortMode === "size") {
        return allTopics
          .slice()
          .sort((a, b) => b.n_papers - a.n_papers)
          .slice(0, count)
          .map((t) => t.topic_id);
      }
      // "rank": trendy topics by trend_rank desc, falling back to size if
      // there aren't enough trendy topics to fill the chart.
      const trendy = allTopics
        .filter((t) => t.is_trendy)
        .sort((a, b) => (b.trend_rank || 0) - (a.trend_rank || 0))
        .map((t) => t.topic_id);
      if (trendy.length >= count) return trendy.slice(0, count);
      const bySize = allTopics.slice().sort((a, b) => b.n_papers - a.n_papers).map((t) => t.topic_id);
      const combined = trendy.concat(bySize.filter((id) => !trendy.includes(id)));
      return combined.slice(0, count);
    }

    function renderTrendChart() {
      const ids = topicIdsForTrendChart();
      trendChartTopicIds = ids;

      Promise.all(
        ids.map((id) => fetch(fillUrl(urls.timeline, { ID: id })).then((r) => r.json()))
      ).then((results) => {
        const traces = results.map((res, i) => ({
          x: res.timeline.map((row) => row.month),
          y: res.timeline.map((row) => row.actual_count),
          name: res.topic.name,
          mode: "lines",
          line: { width: 2, color: LINE_PALETTE[i % LINE_PALETTE.length] },
          hovertemplate: "%{y} mentions<br>%{x}<extra>" + res.topic.name + "</extra>",
        }));
        Plotly.newPlot(
          "trend-chart",
          traces,
          baseLayout({ yaxis: { title: "Mentions / month", gridcolor: "#E8EAED", rangemode: "tozero" } }),
          PLOTLY_CONFIG
        );

        const chartEl = document.getElementById("trend-chart");
        chartEl.removeAllListeners && chartEl.removeAllListeners("plotly_click");
        chartEl.on("plotly_click", (data) => {
          const point = data.points[0];
          const topicId = trendChartTopicIds[point.curveNumber];
          if (topicId != null) {
            window.location.href = fillUrl(urls.topicPage, { ID: topicId });
          }
        });
      });
    }

    // Rebuilds the chip bar and the stacked chart list from
    // `selectedTopicIds`. Called after any change to the selection.
    function renderSelectedTopics() {
      const bar = document.getElementById("selected-topics-bar");
      const chipsEl = document.getElementById("selected-topics-chips");
      const countEl = document.getElementById("selected-topics-count");
      const listEl = document.getElementById("topic-chart-list");
      const emptyEl = document.getElementById("topic-chart-empty");

      countEl.textContent = selectedTopicIds.length;
      bar.hidden = selectedTopicIds.length === 0;

      chipsEl.innerHTML = "";
      selectedTopicIds.forEach((id) => {
        const t = topicsById.get(id);
        if (!t) return;
        const li = document.createElement("li");
        li.className = "selected-topics__chip";
        const label = document.createElement("span");
        label.textContent = (t.is_trendy ? "▲ " : "") + t.name;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "selected-topics__chip-remove";
        remove.setAttribute("aria-label", "Remove " + t.name);
        remove.textContent = "×";
        remove.addEventListener("click", () => toggleSelection(id));
        li.appendChild(label);
        li.appendChild(remove);
        chipsEl.appendChild(li);
      });

      listEl.innerHTML = "";
      emptyEl.hidden = selectedTopicIds.length !== 0;
      selectedTopicIds.forEach((id) => {
        const t = topicsById.get(id);
        if (!t) return;
        const block = document.createElement("div");
        block.className = "topic-chart-block";
        const heading = document.createElement("h3");
        heading.className = "topic-chart-block__title";
        heading.textContent = (t.is_trendy ? "▲ " : "") + t.name;
        const chartDiv = document.createElement("div");
        chartDiv.id = "topic-chart-" + id;
        chartDiv.className = "chart chart--tall";
        const kwDiv = document.createElement("div");
        kwDiv.id = "topic-keywords-" + id;
        kwDiv.className = "keyword-chips";
        block.appendChild(heading);
        block.appendChild(chartDiv);
        block.appendChild(kwDiv);
        listEl.appendChild(block);
        renderOneTopicChart(id);
      });
    }

    function renderOneTopicChart(topicId) {
      const chartId = "topic-chart-" + topicId;
      const kwId = "topic-keywords-" + topicId;
      fetch(fillUrl(urls.timeline, { ID: topicId }))
        .then((r) => r.json())
        .then((res) => {
          // The selection (and so the DOM) may have changed again before
          // this fetch resolved -- bail rather than plot into a stale/gone div.
          if (!document.getElementById(chartId)) return;

          const months = res.timeline.map((row) => row.month);
          const actual = {
            x: months,
            y: res.timeline.map((row) => row.actual_count),
            name: "Actual",
            mode: "lines+markers",
            line: { color: "#2B6E68", width: 2 },
            marker: { size: 5 },
          };
          const predicted = {
            x: months,
            y: res.timeline.map((row) => row.predicted_count),
            name: "Predicted",
            mode: "lines",
            line: { color: "#D98A3D", width: 2, dash: "dot" },
          };
          Plotly.newPlot(
            chartId,
            [actual, predicted],
            baseLayout({ yaxis: { title: "Mentions / month", gridcolor: "#E8EAED", rangemode: "tozero" } }),
            PLOTLY_CONFIG
          );

          document.getElementById(chartId).on("plotly_click", (data) => {
            const month = data.points[0].x;
            goToMonth(urls, topicId, month);
          });

          const kwContainer = document.getElementById(kwId);
          if (kwContainer) {
            kwContainer.innerHTML = "";
            (res.topic.keywords || []).forEach((kw) => {
              const chip = document.createElement("span");
              chip.className = "chip";
              chip.textContent = kw;
              kwContainer.appendChild(chip);
            });
          }
        });
    }
  }

  // ---- Topic detail page --------------------------------------------------

  function initTopicPage() {
    const topic = window.GALENOS_TOPIC;
    const timeline = window.GALENOS_TIMELINE;
    const urls = window.GALENOS.urls;

    const months = timeline.map((row) => row.month);
    const actual = {
      x: months,
      y: timeline.map((row) => row.actual_count),
      name: "Actual",
      mode: "lines+markers",
      line: { color: "#2B6E68", width: 2 },
      marker: { size: 5 },
    };
    const predicted = {
      x: months,
      y: timeline.map((row) => row.predicted_count),
      name: "Predicted",
      mode: "lines",
      line: { color: "#D98A3D", width: 2, dash: "dot" },
    };

    Plotly.newPlot(
      "topic-detail-chart",
      [actual, predicted],
      baseLayout({ yaxis: { title: "Mentions / month", gridcolor: "#E8EAED", rangemode: "tozero" } }),
      PLOTLY_CONFIG
    );

    document.getElementById("topic-detail-chart").on("plotly_click", (data) => {
      const month = data.points[0].x;
      goToMonth(urls, topic.topic_id, month);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!window.GALENOS) return;
    if (typeof Plotly === "undefined") {
      document.querySelectorAll(".chart").forEach((el) => {
        el.innerHTML =
          '<p style="color:#B5423A;font-size:0.85rem;">Chart library failed to load from the CDN ' +
          "(cdn.plot.ly). Check your network/firewall access to that domain, or open the browser " +
          "console for the failed request.</p>";
      });
      return;
    }
    if (window.GALENOS.page === "topic") {
      initTopicPage();
    } else {
      initIndexPage();
      renderEvidenceMap("evidence-map-chart", window.GALENOS_EVIDENCE_MAP);
    }
    renderStudyTypeChart("study-type-chart", window.GALENOS_STUDY_TYPES);
    initSortableTables();
  });
})();
