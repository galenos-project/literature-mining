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
        xaxis: { gridcolor: "#EFEBDE", zeroline: false },
        yaxis: { gridcolor: "#EFEBDE", zeroline: false, rangemode: "tozero" },
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
    let trendChartTopicIds = []; // curveNumber -> topic_id, for click-through
    let trendSortMode = "rank";
    let trendTopicCount = 10;

    fetch(urls.topics)
      .then((r) => r.json())
      .then((topics) => {
        allTopics = topics;
        populateTopicSelect(topics);
        renderTrendChart();
        const first = window.GALENOS.defaultTopicIds[0] || (topics[0] && topics[0].topic_id);
        if (first != null) {
          document.getElementById("topic-select").value = first;
          renderTopicChart(first);
        }
      });

    fetch(urls.paperScatter)
      .then((r) => r.json())
      .then(renderPaperScatter);

    document.getElementById("topic-select").addEventListener("change", (e) => {
      renderTopicChart(e.target.value);
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

    function populateTopicSelect(topics) {
      const select = document.getElementById("topic-select");
      select.innerHTML = "";
      topics.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.topic_id;
        opt.textContent = (t.is_trendy ? "\u25B2 " : "") + t.name;
        select.appendChild(opt);
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
          baseLayout({ yaxis: { title: "Mentions / month", gridcolor: "#EFEBDE", rangemode: "tozero" } }),
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
          marker: { size: 7, color: "#B9B4A5", opacity: 0.65, line: { width: 0.5, color: "#FAF9F4" } },
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
            line: { width: 0.6, color: "#FAF9F4" },
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

    function renderTopicChart(topicId) {
      fetch(fillUrl(urls.timeline, { ID: topicId }))
        .then((r) => r.json())
        .then((res) => {
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
            "topic-chart",
            [actual, predicted],
            baseLayout({ yaxis: { title: "Mentions / month", gridcolor: "#EFEBDE", rangemode: "tozero" } }),
            PLOTLY_CONFIG
          );

          const chartEl = document.getElementById("topic-chart");
          chartEl.removeAllListeners && chartEl.removeAllListeners("plotly_click");
          chartEl.on("plotly_click", (data) => {
            const month = data.points[0].x;
            goToMonth(urls, topicId, month);
          });

          const kwContainer = document.getElementById("topic-keywords");
          kwContainer.innerHTML = "";
          (res.topic.keywords || []).forEach((kw) => {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = kw;
            kwContainer.appendChild(chip);
          });
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
      baseLayout({ yaxis: { title: "Mentions / month", gridcolor: "#EFEBDE", rangemode: "tozero" } }),
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
