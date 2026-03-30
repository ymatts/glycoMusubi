# glycoMusubi Shiny App — Glycan Retrieval Module
# Protein-to-glycan similarity search and retrieval benchmark display.

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
glycan_retrieval_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(
    sidebar = sidebar(
      title = "Retrieval Query",
      width = 300,
      textInput(ns("query_protein"), "UniProt ID",
                placeholder = "e.g. P12345"),
      selectInput(ns("retrieval_mode"), "Retrieval Mode",
                  choices = c("site-level", "protein-level"),
                  selected = "site-level"),
      numericInput(ns("top_k"), "Top-K Results",
                   value = 20, min = 1, max = 100, step = 1),
      actionButton(ns("search"), "Search",
                   class = "btn-primary w-100",
                   icon = icon("sort-amount-down"))
    ),

    uiOutput(ns("results_ui"))
  )
}

# ══════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════
glycan_retrieval_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    # ── Reactive: search results ──
    search_results <- reactiveVal(NULL)
    has_searched <- reactiveVal(FALSE)

    observeEvent(input$search, {
      uid <- trimws(input$query_protein)
      req(nchar(uid) > 0)
      has_searched(TRUE)

      if (nrow(retrieval_rankings) == 0 ||
          !"query_protein" %in% names(retrieval_rankings)) {
        search_results(NULL)
        return()
      }

      results <- retrieval_rankings %>%
        filter(query_protein == uid)

      # Filter by mode if column exists
      if ("mode" %in% names(retrieval_rankings)) {
        results <- results %>% filter(mode == input$retrieval_mode)
      }

      # Take top-k
      if (nrow(results) > 0) {
        results <- results %>%
          arrange(rank) %>%
          head(input$top_k)
      }

      search_results(results)
    })

    # ── Main results UI ──
    output$results_ui <- renderUI({
      ns <- session$ns

      if (!has_searched()) {
        return(tagList(
          benchmark_summary_card(),
          card(
            card_body(
              p(class = "text-muted",
                "Enter a UniProt ID and click Search to retrieve ranked glycan",
                "candidates for the query protein.")
            )
          )
        ))
      }

      results <- search_results()

      if (is.null(results) || nrow(results) == 0) {
        return(tagList(
          card(
            card_header(icon("info-circle"), " No Pre-Computed Results"),
            card_body(
              class = "bg-light",
              p("No pre-computed retrieval rankings are available for this",
                "query protein. The retrieval module uses embedding-based",
                "similarity search over the glycoMusubi knowledge graph."),
              p("Below is a summary of benchmark results from the paper.")
            )
          ),
          benchmark_summary_card()
        ))
      }

      tagList(
        card(
          card_header(paste0("Top-", nrow(results), " Retrieved Glycans")),
          card_body(DTOutput(ns("results_table")))
        ),
        card(
          card_header("Score Distribution"),
          card_body(plotlyOutput(ns("score_plot"), height = "400px"))
        )
      )
    })

    # ── Results table ──
    output$results_table <- renderDT({
      results <- search_results()
      req(!is.null(results), nrow(results) > 0)

      # Enrich with glycan type if available
      display_df <- results %>%
        select(any_of(c("rank", "glycan_id", "score", "query_site")))

      if (nrow(glycans_df) > 0) {
        id_col <- if ("glytoucan_id" %in% names(glycans_df)) "glytoucan_id"
                  else if ("node_id" %in% names(glycans_df)) "node_id"
                  else NULL

        if (!is.null(id_col)) {
          type_col <- intersect(c("glycan_type", "type", "class"), names(glycans_df))
          if (length(type_col) > 0) {
            glycan_types <- glycans_df %>%
              select(all_of(c(id_col, type_col[1]))) %>%
              rename(glycan_id = !!id_col, glycan_type = !!type_col[1])

            display_df <- display_df %>%
              left_join(glycan_types, by = "glycan_id")
          }
        }
      }

      # Rename for display
      col_names <- c(
        rank = "Rank",
        glycan_id = "Glycan ID",
        score = "Similarity Score",
        query_site = "Query Site",
        glycan_type = "Glycan Type"
      )
      display_df <- display_df %>% rename(any_of(col_names))

      datatable(display_df,
                rownames = FALSE,
                options = list(pageLength = 20, scrollX = TRUE,
                               order = list(list(0, "asc"))),
                class = "compact stripe hover") %>%
        formatRound(columns = "Similarity Score", digits = 4)
    })

    # ── Score distribution plot ──
    output$score_plot <- renderPlotly({
      results <- search_results()
      req(!is.null(results), nrow(results) > 0)

      df <- results %>%
        arrange(desc(rank)) %>%
        mutate(label = truncate_str(glycan_id, 30))

      plot_ly(df,
              y = ~reorder(label, -rank),
              x = ~score,
              type = "bar",
              orientation = "h",
              marker = list(color = "#2b6a99"),
              text = ~paste0(glycan_id, ": ", fmt_metric(score, 4)),
              hoverinfo = "text") %>%
        layout(
          xaxis = list(title = "Similarity Score"),
          yaxis = list(title = ""),
          margin = list(l = 150),
          showlegend = FALSE
        )
    })
  })
}

# ══════════════════════════════════════════════════════════════
# Static benchmark summary (shown when no pre-computed results)
# ══════════════════════════════════════════════════════════════
benchmark_summary_card <- function() {
  # Hardcoded benchmark results from the paper
  benchmark_df <- data.frame(
    Level = c(
      "Site-level", "Site-level", "Site-level",
      "Protein-level", "Protein-level", "Protein-level"
    ),
    Metric = c(
      "MRR", "Hits@10", "Hits@50",
      "MRR", "Hits@10", "Hits@50"
    ),
    Value = c(
      "0.066", "0.128", "0.251",
      "0.118", "0.215", "0.389"
    ),
    stringsAsFactors = FALSE
  )

  hierarchy_df <- data.frame(
    `Prediction Target` = c(
      "Monosaccharide composition",
      "Core structure class",
      "Structural cluster",
      "Exact glycan (GlyTouCan ID)"
    ),
    `Site-level MRR` = c("0.312", "0.198", "0.066", "0.031"),
    `Protein-level MRR` = c("0.458", "0.301", "0.118", "0.052"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  card(
    card_header("Retrieval Benchmark Summary"),
    card_body(
      p("The glycan retrieval task ranks candidate glycans for a query",
        "protein (or protein-site pair) using learned KG embeddings.",
        "Results below are from the glycoMusubi paper evaluation."),
      layout_columns(
        col_widths = c(6, 6),
        card(
          card_header("Overall Metrics"),
          card_body(
            tags$table(
              class = "table table-sm table-striped",
              tags$thead(
                tags$tr(
                  tags$th("Level"),
                  tags$th("Metric"),
                  tags$th("Value")
                )
              ),
              tags$tbody(
                lapply(seq_len(nrow(benchmark_df)), function(i) {
                  tags$tr(
                    tags$td(benchmark_df$Level[i]),
                    tags$td(benchmark_df$Metric[i]),
                    tags$td(tags$code(benchmark_df$Value[i]))
                  )
                })
              )
            )
          )
        ),
        card(
          card_header("Hierarchical Prediction Levels"),
          card_body(
            tags$table(
              class = "table table-sm table-striped",
              tags$thead(
                tags$tr(
                  tags$th("Prediction Target"),
                  tags$th("Site-level MRR"),
                  tags$th("Protein-level MRR")
                )
              ),
              tags$tbody(
                lapply(seq_len(nrow(hierarchy_df)), function(i) {
                  tags$tr(
                    tags$td(hierarchy_df$`Prediction Target`[i]),
                    tags$td(tags$code(hierarchy_df$`Site-level MRR`[i])),
                    tags$td(tags$code(hierarchy_df$`Protein-level MRR`[i]))
                  )
                })
              )
            ),
            tags$hr(),
            tags$small(class = "text-muted",
              "MRR = Mean Reciprocal Rank. Higher values indicate better",
              "retrieval performance. The determination boundary is at the",
              "structural-class level.")
          )
        )
      )
    )
  )
}
