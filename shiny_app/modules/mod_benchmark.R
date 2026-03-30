# glycoMusubi Shiny App — Benchmark Dashboard Module
# Hierarchical glycan prediction results and experiment comparison.

# ── UI ──
benchmark_ui <- function(id) {
  ns <- NS(id)

  tagList(
    # ── Top-level key metrics ──
    layout_columns(
      fill = FALSE,
      col_widths = c(3, 3, 3, 3),
      value_box_ui(0.924, "N-linked F1", icon = "bullseye", color = "#2b6a99"),
      value_box_ui(0.786, "Cluster H@5", icon = "layer-group", color = "#3a7db5"),
      value_box_ui(0.066, "Exact MRR", icon = "sort-amount-down", color = "#c04e3f"),
      value_box_ui("10x", "Determination Gap", icon = "arrows-alt-v", color = "#d4764a")
    ),

    # ── Hierarchical benchmark results ──
    layout_columns(
      col_widths = c(6, 6),
      card(
        card_header("Hierarchical Benchmark Results (Table 2)"),
        card_body(DTOutput(ns("dt_hierarchy")))
      ),
      card(
        card_header("6-Task Hierarchy Performance"),
        card_body(plotlyOutput(ns("plot_hierarchy"), height = "420px"))
      )
    ),

    # ── Experiment comparison ──
    card(
      card_header("Experiment Comparison"),
      card_body(
        selectizeInput(ns("sel_experiments"), "Select Experiments",
                       choices = names(all_metrics),
                       multiple = TRUE,
                       options = list(maxItems = 10,
                                      placeholder = "Choose experiments to compare...")),
        DTOutput(ns("dt_comparison"))
      )
    ),

    # ── Training history ──
    card(
      card_header("Training History"),
      card_body(
        selectInput(ns("sel_train_exp"), "Experiment",
                    choices = names(all_metrics)),
        plotlyOutput(ns("plot_train_history"), height = "350px")
      )
    )
  )
}

# ── Server ──
benchmark_server <- function(id) {

  moduleServer(id, function(input, output, session) {

    # ══════════════════════════════════════════════════════════════
    # Hardcoded Table 2 — Hierarchical benchmark results
    # ══════════════════════════════════════════════════════════════
    hierarchy_df <- data.frame(
      Task = c("N-linked (binary)", "O-linked (binary)",
               "Structural class (6-way)", "Cluster (121-way)",
               "Motif composition", "Exact structure"),
      Level = c(1, 1, 2, 3, 4, 5),
      Metric = c("F1", "F1", "Macro-F1", "Hits@5", "Jaccard", "MRR"),
      Score = c(0.924, 0.889, 0.841, 0.786, 0.312, 0.066),
      Baseline = c(0.510, 0.500, 0.167, 0.008, 0.041, 0.002),
      Fold_over_BL = c(1.8, 1.8, 5.0, 98.3, 7.6, 33.0),
      stringsAsFactors = FALSE
    )

    # ── Hierarchy table ──
    output$dt_hierarchy <- DT::renderDT({
      df <- hierarchy_df
      df$Score <- sapply(df$Score, fmt_metric)
      df$Baseline <- sapply(df$Baseline, fmt_metric)
      df$Fold_over_BL <- paste0(hierarchy_df$Fold_over_BL, "x")
      names(df)[names(df) == "Fold_over_BL"] <- "Fold over BL"

      datatable(df, rownames = FALSE,
                options = list(dom = "t", paging = FALSE, ordering = FALSE),
                class = "compact stripe") %>%
        formatStyle("Score",
                    backgroundColor = styleInterval(
                      c(0.3, 0.7),
                      c("#f8d7da", "#fff3cd", "#d4edda")
                    ))
    })

    # ── Hierarchy bar chart ──
    output$plot_hierarchy <- renderPlotly({
      df <- hierarchy_df

      # Determination boundary between Level 3 (Cluster) and Level 4 (Motif)
      plot_ly() %>%
        add_trace(
          data = df, x = ~Task, y = ~Score,
          type = "bar", name = "glycoMusubi",
          marker = list(color = "#2b6a99"),
          text = ~paste0(Score),
          hovertemplate = "%{x}<br>Score: %{y:.3f}<extra>glycoMusubi</extra>"
        ) %>%
        add_trace(
          data = df, x = ~Task, y = ~Baseline,
          type = "bar", name = "Baseline",
          marker = list(color = "#cccccc"),
          hovertemplate = "%{x}<br>Score: %{y:.3f}<extra>Baseline</extra>"
        ) %>%
        layout(
          barmode = "group",
          xaxis = list(title = "", tickangle = -30,
                       categoryorder = "array",
                       categoryarray = df$Task),
          yaxis = list(title = "Score", range = c(0, 1.05)),
          margin = list(b = 140),
          legend = list(orientation = "h", y = 1.08),
          shapes = list(
            list(
              type = "line",
              x0 = 3.5, x1 = 3.5,
              y0 = 0, y1 = 1.0,
              line = list(color = "#c04e3f", width = 2, dash = "dash")
            )
          ),
          annotations = list(
            list(
              x = 3.5, y = 1.02,
              text = "Determination\nboundary",
              showarrow = FALSE,
              font = list(color = "#c04e3f", size = 11),
              xanchor = "center"
            )
          )
        )
    })

    # ══════════════════════════════════════════════════════════════
    # Experiment comparison
    # ══════════════════════════════════════════════════════════════
    output$dt_comparison <- DT::renderDT({
      req(length(input$sel_experiments) > 0)

      sel <- input$sel_experiments
      metric_names <- c("mrr", "hits_at_1", "hits_at_5", "hits_at_10",
                        "f1", "macro_f1", "accuracy", "precision", "recall",
                        "jaccard", "auc")

      rows <- lapply(sel, function(exp_name) {
        m <- all_metrics[[exp_name]]
        if (is.null(m)) return(NULL)

        vals <- sapply(metric_names, function(mn) {
          v <- m[[mn]]
          if (is.null(v)) NA_real_ else as.numeric(v)
        })
        c(experiment = exp_name, vals)
      })

      rows <- rows[!sapply(rows, is.null)]
      if (length(rows) == 0) {
        return(datatable(data.frame(Message = "No metrics found for selected experiments.")))
      }

      df <- do.call(rbind, lapply(rows, function(r) as.data.frame(t(r), stringsAsFactors = FALSE)))
      names(df) <- c("Experiment", metric_names)

      # Convert to numeric
      for (col in metric_names) {
        df[[col]] <- as.numeric(df[[col]])
        df[[col]] <- sapply(df[[col]], fmt_metric)
      }

      # Clean column names for display
      display_names <- c("Experiment", "MRR", "H@1", "H@5", "H@10",
                         "F1", "Macro-F1", "Accuracy", "Precision", "Recall",
                         "Jaccard", "AUC")
      names(df) <- display_names

      datatable(df, rownames = FALSE, filter = "none",
                options = list(scrollX = TRUE, dom = "t", paging = FALSE),
                class = "compact stripe")
    })

    # ══════════════════════════════════════════════════════════════
    # Training history
    # ══════════════════════════════════════════════════════════════
    output$plot_train_history <- renderPlotly({
      req(input$sel_train_exp)
      exp_name <- input$sel_train_exp

      # Look for train_history.json in the experiment directory
      hist_path <- file.path(EXP_DIR, exp_name, "train_history.json")

      if (!file.exists(hist_path)) {
        return(
          plotly_empty() %>%
            layout(title = list(
              text = paste0("No training history found for '", exp_name, "'"),
              font = list(size = 14)
            ))
        )
      }

      hist <- safe_read_json(hist_path)

      # Expect a list/data.frame with columns: epoch, train_loss, val_loss,
      # and optionally val_metric
      if (is.null(hist) || length(hist) == 0) {
        return(plotly_empty() %>%
                 layout(title = list(text = "Training history file is empty")))
      }

      df <- as.data.frame(hist)

      # Build traces dynamically based on available columns
      p <- plot_ly(df, x = ~epoch)

      if ("train_loss" %in% names(df)) {
        p <- p %>%
          add_trace(y = ~train_loss, name = "Train Loss",
                    type = "scatter", mode = "lines+markers",
                    line = list(color = "#2b6a99"),
                    marker = list(size = 4))
      }

      if ("val_loss" %in% names(df)) {
        p <- p %>%
          add_trace(y = ~val_loss, name = "Val Loss",
                    type = "scatter", mode = "lines+markers",
                    line = list(color = "#c04e3f"),
                    marker = list(size = 4))
      }

      if ("val_metric" %in% names(df)) {
        p <- p %>%
          add_trace(y = ~val_metric, name = "Val Metric",
                    type = "scatter", mode = "lines+markers",
                    line = list(color = "#3a7db5", dash = "dot"),
                    marker = list(size = 4),
                    yaxis = "y2")
      }

      p %>%
        layout(
          xaxis = list(title = "Epoch"),
          yaxis = list(title = "Loss"),
          yaxis2 = list(title = "Metric", overlaying = "y", side = "right"),
          legend = list(orientation = "h", y = 1.1),
          margin = list(r = 60)
        )
    })
  })
}
