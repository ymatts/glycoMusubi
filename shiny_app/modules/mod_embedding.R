# glycoMusubi Shiny App — Embedding Explorer Module
# Visualize UMAP projections of glycan KG embeddings and protein ESM-2 embeddings.

# ── UI ──
embedding_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(
    sidebar = sidebar(
      title = "Embedding Settings",
      width = 300,
      radioButtons(ns("embedding_type"), "Embedding Type",
                   choices = c("Glycan (KG embeddings)" = "glycan",
                               "Protein (ESM-2)" = "protein"),
                   selected = "glycan"),
      selectInput(ns("color_by"), "Color By",
                  choices = c("cluster", "glycan_type")),
      textInput(ns("highlight_ids"), "Highlight IDs",
                placeholder = "e.g. G00055MO, G12345AB"),
      sliderInput(ns("point_size"), "Point Size",
                  min = 1, max = 5, value = 2, step = 0.5),
      checkboxInput(ns("show_labels"), "Show Labels", value = FALSE),
      actionButton(ns("btn_plot"), "Plot",
                   class = "btn-primary w-100",
                   icon = icon("braille"))
    ),

    # ── UMAP scatter plot ──
    card(
      card_header("UMAP Projection"),
      card_body(plotlyOutput(ns("umap_plot"), height = "600px"))
    ),

    # ── Statistics row ──
    layout_columns(
      fill = FALSE,
      col_widths = c(4, 4, 4),
      uiOutput(ns("vb_n_points")),
      uiOutput(ns("vb_n_clusters")),
      uiOutput(ns("vb_variance_note"))
    ),

    # ── Detail table for clicked / highlighted points ──
    card(
      card_header("Point Details"),
      card_body(DTOutput(ns("dt_details")))
    )
  )
}

# ── Server ──
embedding_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    # Update color_by choices when embedding type changes
    observeEvent(input$embedding_type, {
      if (input$embedding_type == "glycan") {
        updateSelectInput(session, "color_by",
                          choices = c("cluster", "glycan_type"),
                          selected = "cluster")
      } else {
        updateSelectInput(session, "color_by",
                          choices = c("none"),
                          selected = "none")
      }
    })

    # Generate demo data for when real UMAP coordinates are not available
    demo_data <- reactive({
      set.seed(42)
      n <- 500
      cluster <- sample(1:5, n, replace = TRUE)
      centers <- matrix(c(-5, -5, 5, -5, 0, 5, -5, 5, 5, 5),
                        ncol = 2, byrow = TRUE)
      x <- centers[cluster, 1] + rnorm(n, sd = 1.2)
      y <- centers[cluster, 2] + rnorm(n, sd = 1.2)
      data.frame(
        id = paste0("DEMO_", sprintf("%04d", seq_len(n))),
        umap_x = x,
        umap_y = y,
        cluster = as.character(cluster),
        glycan_type = sample(c("N-linked", "O-linked", "GAG", "Lipid", "Other"),
                             n, replace = TRUE),
        stringsAsFactors = FALSE
      )
    })

    # Resolve whether real data exists
    has_real_data <- reactive({
      if (input$embedding_type == "glycan") {
        nrow(umap_glycans) > 0
      } else {
        nrow(umap_proteins) > 0
      }
    })

    # Active dataset (real or demo)
    active_data <- reactive({
      if (input$embedding_type == "glycan") {
        if (nrow(umap_glycans) > 0) umap_glycans else demo_data()
      } else {
        if (nrow(umap_proteins) > 0) {
          umap_proteins
        } else {
          # Demo data without glycan-specific columns
          dd <- demo_data()
          dd[, c("id", "umap_x", "umap_y")]
        }
      }
    })

    # Parse highlighted IDs
    highlight_set <- reactive({
      ids_raw <- trimws(input$highlight_ids)
      if (nchar(ids_raw) == 0) return(character(0))
      ids <- trimws(unlist(strsplit(ids_raw, ",")))
      ids[nchar(ids) > 0]
    })

    # Reactive: plot data (triggered by button)
    plot_data <- eventReactive(input$btn_plot, {
      df <- active_data()
      color_col <- input$color_by
      hl_ids <- highlight_set()

      # Determine color column
      if (color_col != "none" && color_col %in% names(df)) {
        df$color_group <- as.character(df[[color_col]])
      } else {
        df$color_group <- "all"
      }

      # Mark highlighted points
      df$highlighted <- df$id %in% hl_ids
      df$display_size <- ifelse(df$highlighted,
                                input$point_size * 2.5,
                                input$point_size)

      # Hover text
      df$hover_text <- paste0("<b>", df$id, "</b>",
                              if ("cluster" %in% names(df))
                                paste0("<br>Cluster: ", df$cluster) else "",
                              if ("glycan_type" %in% names(df))
                                paste0("<br>Type: ", df$glycan_type) else "",
                              "<br>UMAP: (", round(df$umap_x, 2),
                              ", ", round(df$umap_y, 2), ")")

      df
    }, ignoreNULL = FALSE)

    # ── UMAP plot ──
    output$umap_plot <- renderPlotly({
      df <- plot_data()
      is_real <- has_real_data()

      # Build base scatter
      p <- plot_ly(source = session$ns("umap_src")) %>%
        config(displayModeBar = TRUE)

      # Plot non-highlighted points by color group
      groups <- unique(df$color_group[!df$highlighted])
      for (grp in groups) {
        sub <- df[!df$highlighted & df$color_group == grp, ]
        if (nrow(sub) == 0) next
        p <- p %>%
          add_trace(
            data = sub,
            x = ~umap_x, y = ~umap_y,
            type = "scattergl", mode = if (input$show_labels) "markers+text" else "markers",
            marker = list(size = input$point_size * 3,
                          opacity = 0.7),
            text = if (input$show_labels) ~id else ~hover_text,
            textposition = "top center",
            textfont = list(size = 8),
            hoverinfo = "text",
            hovertext = ~hover_text,
            name = grp,
            key = ~id
          )
      }

      # Plot highlighted points with black border
      hl <- df[df$highlighted, ]
      if (nrow(hl) > 0) {
        p <- p %>%
          add_trace(
            data = hl,
            x = ~umap_x, y = ~umap_y,
            type = "scattergl", mode = if (input$show_labels) "markers+text" else "markers",
            marker = list(size = input$point_size * 6,
                          opacity = 1,
                          line = list(color = "black", width = 2)),
            text = if (input$show_labels) ~id else ~hover_text,
            textposition = "top center",
            textfont = list(size = 9, color = "black"),
            hoverinfo = "text",
            hovertext = ~hover_text,
            name = "Highlighted",
            key = ~id
          )
      }

      # Layout
      title_suffix <- if (!is_real) " (Demo Data)" else ""
      p <- p %>%
        layout(
          title = list(
            text = paste0("UMAP — ",
                          if (input$embedding_type == "glycan") "Glycan KG Embeddings"
                          else "Protein ESM-2 Embeddings",
                          title_suffix),
            font = list(size = 14)
          ),
          xaxis = list(title = "UMAP 1", zeroline = FALSE),
          yaxis = list(title = "UMAP 2", zeroline = FALSE),
          legend = list(title = list(text = input$color_by)),
          hovermode = "closest",
          dragmode = "zoom"
        )

      # Add annotation when using demo data
      if (!is_real) {
        p <- p %>%
          layout(annotations = list(
            list(
              text = paste0("Demo data shown. Run prepare_shiny_data.py to generate",
                            " real UMAP coordinates."),
              xref = "paper", yref = "paper",
              x = 0.5, y = -0.12,
              showarrow = FALSE,
              font = list(size = 11, color = "#c04e3f")
            )
          ))
      }

      p
    })

    # ── Value boxes ──
    output$vb_n_points <- renderUI({
      df <- plot_data()
      value_box_ui(nrow(df), "Points Plotted",
                   icon = "circle", color = "#2b6a99")
    })

    output$vb_n_clusters <- renderUI({
      df <- plot_data()
      n_clust <- if ("cluster" %in% names(df)) {
        length(unique(df$cluster))
      } else {
        "N/A"
      }
      value_box_ui(n_clust, "Clusters",
                   icon = "layer-group", color = "#3a7db5")
    })

    output$vb_variance_note <- renderUI({
      is_real <- has_real_data()
      label <- if (is_real) "Pre-computed UMAP" else "Demo (random) data"
      value_box_ui(if (is_real) "Real" else "Demo", label,
                   icon = "info-circle", color = "#4a9fd4")
    })

    # ── Detail table: clicked + highlighted points ──
    selected_ids <- reactiveVal(character(0))

    # Capture plotly click events
    observeEvent(event_data("plotly_click", source = session$ns("umap_src")), {
      click <- event_data("plotly_click", source = session$ns("umap_src"))
      if (!is.null(click) && "key" %in% names(click)) {
        current <- selected_ids()
        new_id <- as.character(click$key)
        # Toggle selection: add if absent, remove if present
        if (new_id %in% current) {
          selected_ids(setdiff(current, new_id))
        } else {
          selected_ids(c(current, new_id))
        }
      }
    })

    output$dt_details <- DT::renderDT({
      df <- plot_data()
      hl_ids <- highlight_set()
      sel_ids <- selected_ids()

      show_ids <- unique(c(hl_ids, sel_ids))
      if (length(show_ids) == 0) {
        detail <- df[FALSE, ]
      } else {
        detail <- df[df$id %in% show_ids, ]
      }

      # Select display columns
      display_cols <- intersect(
        c("id", "umap_x", "umap_y", "cluster", "glycan_type", "color_group"),
        names(detail)
      )
      detail <- detail[, display_cols, drop = FALSE]

      # Round coordinates
      if ("umap_x" %in% names(detail)) detail$umap_x <- round(detail$umap_x, 4)
      if ("umap_y" %in% names(detail)) detail$umap_y <- round(detail$umap_y, 4)

      datatable(detail, rownames = FALSE, filter = "top",
                options = list(pageLength = 10, scrollX = TRUE))
    })
  })
}
