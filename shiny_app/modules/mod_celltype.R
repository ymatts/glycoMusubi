# glycoMusubi Shiny App — Cell-type Expression Module
# Explore glycosylation enzyme expression across 748 cell types.

# ── UI ──
celltype_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(
    sidebar = sidebar(
      title = "Expression Filter",
      width = 320,
      selectizeInput(ns("cell_types"), "Cell Types",
                     choices = NULL, multiple = TRUE,
                     options = list(maxItems = 20,
                                    placeholder = "Search cell types...")),
      selectizeInput(ns("enzymes"), "Enzymes / Genes",
                     choices = NULL, multiple = TRUE,
                     options = list(maxItems = 30,
                                    placeholder = "Search enzymes...")),
      radioButtons(ns("plot_type"), "Plot Type",
                   choices = c("Heatmap", "Bar chart", "Correlation"),
                   selected = "Heatmap"),
      actionButton(ns("btn_visualize"), "Visualize",
                   class = "btn-primary w-100",
                   icon = icon("chart-area")),

      tags$hr(),
      tags$h6("Predefined Sets"),
      actionButton(ns("btn_core"), "Core pathway enzymes",
                   class = "btn-outline-secondary btn-sm w-100 mb-2"),
      actionButton(ns("btn_terminal"), "Terminal processing",
                   class = "btn-outline-secondary btn-sm w-100")
    ),

    # ── Value boxes ──
    layout_columns(
      fill = FALSE,
      col_widths = c(4, 4, 4),
      uiOutput(ns("vb_total_celltypes")),
      uiOutput(ns("vb_total_enzymes")),
      uiOutput(ns("vb_selected_celltypes"))
    ),

    # ── Main plot ──
    card(
      card_header("Expression Visualization"),
      card_body(plotlyOutput(ns("main_plot"), height = "500px"))
    ),

    # ── Expression table ──
    card(
      card_header("Expression Values"),
      card_body(DTOutput(ns("expr_table")))
    )
  )
}

# ── Server ──
celltype_server <- function(id) {

  moduleServer(id, function(input, output, session) {

    # Guard: handle empty celltype_expr
    if (is.null(celltype_expr) || nrow(celltype_expr) == 0) {
      output$main_plot <- renderPlotly({
        plotly_empty() %>%
          layout(title = list(text = "No cell-type expression data available"))
      })
      output$expr_table <- DT::renderDT({ datatable(data.frame()) })
      return()
    }

    # ── Derive column sets ──
    enzyme_cols <- setdiff(names(celltype_expr), "cell_type")
    all_cell_types <- sort(unique(celltype_expr$cell_type))

    # ── Server-side selectize updates ──
    updateSelectizeInput(session, "cell_types",
                         choices = all_cell_types,
                         server = TRUE)
    updateSelectizeInput(session, "enzymes",
                         choices = enzyme_cols,
                         server = TRUE)

    # ── Predefined enzyme sets ──
    core_enzymes <- c("MGAT1", "MGAT2", "MAN1A1", "MAN1A2",
                      "MAN2A1", "MAN2A2", "GANAB", "MOGS")
    terminal_enzymes <- c(
      paste0("ST3GAL", 1:6),
      paste0("FUT", 1:8),
      paste0("B4GALT", 1:4)
    )

    observeEvent(input$btn_core, {
      avail <- intersect(core_enzymes, enzyme_cols)
      updateSelectizeInput(session, "enzymes", selected = avail)
    })

    observeEvent(input$btn_terminal, {
      avail <- intersect(terminal_enzymes, enzyme_cols)
      updateSelectizeInput(session, "enzymes", selected = avail)
    })

    # ── Value boxes ──
    output$vb_total_celltypes <- renderUI({
      value_box_ui(length(all_cell_types), "Total Cell Types",
                   icon = "microscope", color = "#2b6a99")
    })

    output$vb_total_enzymes <- renderUI({
      value_box_ui(length(enzyme_cols), "Total Enzymes",
                   icon = "flask", color = "#3a7db5")
    })

    output$vb_selected_celltypes <- renderUI({
      n <- length(input$cell_types)
      value_box_ui(n, "Selected Cell Types",
                   icon = "check-circle", color = "#4a9fd4")
    })

    # ── Reactive: filtered expression matrix ──
    expr_filtered <- eventReactive(input$btn_visualize, {
      sel_ct <- input$cell_types
      sel_enz <- input$enzymes

      validate(
        need(length(sel_ct) > 0, "Please select at least one cell type."),
        need(length(sel_enz) > 0, "Please select at least one enzyme.")
      )

      celltype_expr %>%
        filter(cell_type %in% sel_ct) %>%
        select(cell_type, all_of(sel_enz))
    })

    # ── Main plot ──
    output$main_plot <- renderPlotly({
      df <- expr_filtered()

      switch(input$plot_type,
        "Heatmap" = {
          mat <- as.matrix(df[, -1, drop = FALSE])
          rownames(mat) <- df$cell_type

          plot_ly(
            z = mat,
            x = colnames(mat),
            y = rownames(mat),
            type = "heatmap",
            colorscale = "Viridis",
            hovertemplate = "Cell type: %{y}<br>Enzyme: %{x}<br>Expression: %{z:.3f}<extra></extra>"
          ) %>%
            layout(
              xaxis = list(title = "", tickangle = -45),
              yaxis = list(title = ""),
              margin = list(b = 120, l = 150)
            )
        },

        "Bar chart" = {
          df_long <- df %>%
            pivot_longer(-cell_type, names_to = "enzyme", values_to = "expression")

          plot_ly(df_long,
                  x = ~enzyme, y = ~expression, color = ~cell_type,
                  type = "bar",
                  hovertemplate = "%{x}: %{y:.3f}<extra>%{fullData.name}</extra>") %>%
            layout(
              barmode = "group",
              xaxis = list(title = "", tickangle = -45),
              yaxis = list(title = "Expression"),
              margin = list(b = 120),
              legend = list(title = list(text = "Cell Type"))
            )
        },

        "Correlation" = {
          sel_enz <- input$enzymes
          validate(
            need(length(sel_enz) >= 2,
                 "Correlation plot requires at least 2 enzymes selected.")
          )

          # Use all cell types for the scatter (not just selected)
          enz_x <- sel_enz[1]
          enz_y <- sel_enz[2]
          df_all <- celltype_expr %>% select(cell_type, all_of(c(enz_x, enz_y)))

          # Highlight selected cell types
          df_all$selected <- df_all$cell_type %in% input$cell_types

          plot_ly(df_all,
                  x = as.formula(paste0("~`", enz_x, "`")),
                  y = as.formula(paste0("~`", enz_y, "`")),
                  color = ~selected,
                  colors = c("FALSE" = "#cccccc", "TRUE" = "#2b6a99"),
                  type = "scatter", mode = "markers",
                  text = ~cell_type,
                  hovertemplate = "%{text}<br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
                  marker = list(size = 8, opacity = 0.7)) %>%
            layout(
              xaxis = list(title = enz_x),
              yaxis = list(title = enz_y),
              showlegend = FALSE
            )
        }
      )
    })

    # ── Expression table ──
    output$expr_table <- DT::renderDT({
      df <- expr_filtered()
      # Round numeric columns for display
      num_cols <- setdiff(names(df), "cell_type")
      df[num_cols] <- lapply(df[num_cols], round, digits = 4)

      datatable(df, rownames = FALSE, filter = "top",
                options = list(pageLength = 15, scrollX = TRUE,
                               dom = "Bfrtip"))
    })
  })
}
