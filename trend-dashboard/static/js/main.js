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

    fetch(urls.paperScatter)
      .then((r) => r.json())
      .then(renderPaperScatter);

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

    function renderPaperScatter(points) {
      const traces = [];

      const greyPoints = points.filter((p) => !p.is_trendy);
      if (greyPoints.length) {
        traces.push({
          x: greyPoints.map((p) => p.embed_x),
          y: greyPoints.map((p) => p.embed_y),
          text: greyPoints.map((p) => `${p.title}<br><i>${p.topic_name}</i>`),
          customdata: greyPoints.map((p) => p.topic_id),
          mode: "markers",
          name: "Established",
          marker: { size: 7, color: "#B7BBC2", opacity: 0.65, line: { width: 0.5, color: "#F3F4F6" } },
          hovertemplate: "%{text}<extra></extra>",
        });
      }

      const trendyTopicIds = [...new Set(points.filter((p) => p.is_trendy).map((p) => p.topic_id))];
      trendyTopicIds.forEach((tid, i) => {
        const pts = points.filter((p) => p.topic_id === tid);
        traces.push({
          x: pts.map((p) => p.embed_x),
          y: pts.map((p) => p.embed_y),
          text: pts.map((p) => `${p.title}<br><i>${p.topic_name}</i>`),
          customdata: pts.map((p) => p.topic_id),
          mode: "markers",
          name: pts[0].topic_name,
          marker: {
            size: 9,
            color: LINE_PALETTE[i % LINE_PALETTE.length],
            opacity: 0.9,
            line: { width: 0.6, color: "#F3F4F6" },
          },
          hovertemplate: "%{text}<extra></extra>",
        });
      });

      Plotly.newPlot(
        "scatter-chart",
        traces,
        baseLayout({
          xaxis: { visible: false },
          yaxis: { visible: false, rangemode: "normal" },
          legend: { orientation: "h", y: -0.18 },
        }),
        PLOTLY_CONFIG
      );

      document.getElementById("scatter-chart").on("plotly_click", (data) => {
        const point = data.points[0];
        window.location.href = fillUrl(urls.topicPage, { ID: point.customdata });
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
    }
  });
})();
