# glycoMusubi Shiny App — Main Application
# Interactive web resource for the glycoMusubi knowledge graph and benchmarks.

source("global.R")

# ── Source all modules ──
module_files <- list.files("modules", pattern = "\\.R$", full.names = TRUE)
for (f in module_files) source(f)

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
ui <- page_navbar(
  title = tags$span(
    tags$img(src = "https://img.shields.io/badge/Glyco--KG-Explorer-2b6a99",
             height = "20px", style = "margin-right:8px;vertical-align:middle;"),
    "glycoMusubi"
  ),
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#2b6a99",
    "navbar-bg" = "#ffffff"
  ),
  header = tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "style.css")
  ),

  # ── Tab 1: Overview ──
  nav_panel(
    title = "Overview",
    icon = icon("home"),
    layout_columns(
      fill = FALSE,
      col_widths = c(3, 3, 3, 3),
      value_box_ui(kg_stats$n_nodes, "Nodes", "project-diagram"),
      value_box_ui(kg_stats$n_edges, "Edges", "share-alt"),
      value_box_ui(kg_stats$n_node_types, "Entity Types", "layer-group"),
      value_box_ui(kg_stats$n_relation_types, "Relation Types", "arrows-alt-h")
    ),
    layout_columns(
      col_widths = c(6, 6),
      card(
        card_header("Node Type Distribution"),
        card_body(plotlyOutput("overview_node_plot", height = "350px"))
      ),
      card(
        card_header("Relation Type Distribution"),
        card_body(plotlyOutput("overview_edge_plot", height = "350px"))
      )
    ),
    card(
      card_header("About glycoMusubi"),
      card_body(
        p("glycoMusubi is an integrated knowledge graph for glycobiology,",
          "spanning 78,263 nodes (10 entity types) and 2.5 million edges",
          "(14 relation types) from six public databases."),
        p("This application provides interactive exploration of the knowledge",
          "graph, hierarchical glycan prediction benchmarks, cell-type enzyme",
          "expression analysis, and embedding-based search tools."),
        tags$ul(
          tags$li(tags$b("KG Explorer"), " — Search and visualize the knowledge graph"),
          tags$li(tags$b("Glycan Predictor"), " — N-linked site and structure class prediction"),
          tags$li(tags$b("Glycan Retrieval"), " — Protein-to-glycan similarity search"),
          tags$li(tags$b("Cell-type Expression"), " — Enzyme expression across 748 cell types"),
          tags$li(tags$b("Benchmarks"), " — Performance dashboard for 33 experiments"),
          tags$li(tags$b("Embedding Explorer"), " — UMAP visualization of learned representations"),
          tags$li(tags$b("Hierarchy Browser"), " — Glycan structural taxonomy tree")
        ),
        tags$hr(),
        tags$small(
          "Matsui Y. (2026) Hierarchical glycan prediction from protein",
          "sequence reveals a determination boundary at the structural-class level.",
          tags$i("Bioinformatics"), ".",
          tags$a(href = "https://github.com/ymatts/glycoMusubi",
                 "GitHub", target = "_blank")
        )
      )
    )
  ),

  # ── Tab 2: KG Explorer ──
  nav_panel(
    title = "KG Explorer",
    icon = icon("project-diagram"),
    kg_explorer_ui("kg_explorer")
  ),

  # ── Tab 3: Glycan Predictor ──
  nav_panel(
    title = "Glycan Predictor",
    icon = icon("search-plus"),
    glycan_predictor_ui("glycan_predictor")
  ),

  # ── Tab 4: Glycan Retrieval ──
  nav_panel(
    title = "Glycan Retrieval",
    icon = icon("sort-amount-down"),
    glycan_retrieval_ui("glycan_retrieval")
  ),

  # ── Tab 5: Cell-type Expression ──
  nav_panel(
    title = "Cell-type Expression",
    icon = icon("microscope"),
    celltype_ui("celltype")
  ),

  # ── Tab 6: Benchmarks ──
  nav_panel(
    title = "Benchmarks",
    icon = icon("chart-bar"),
    benchmark_ui("benchmark")
  ),

  # ── Tab 7: Embedding Explorer ──
  nav_panel(
    title = "Embedding Explorer",
    icon = icon("braille"),
    embedding_ui("embedding")
  ),

  # ── Tab 8: Hierarchy Browser ──
  nav_panel(
    title = "Hierarchy Browser",
    icon = icon("sitemap"),
    hierarchy_ui("hierarchy")
  ),

  # ── Footer nav ──
  nav_spacer(),
  nav_item(
    tags$a(icon("github"), "Source",
           href = "https://github.com/ymatts/glycoMusubi",
           target = "_blank", class = "btn btn-sm btn-outline-secondary")
  )
)

# ══════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════
server <- function(input, output, session) {

  # ── Overview plots ──
  output$overview_node_plot <- renderPlotly({
    df <- node_type_summary
    colors <- node_type_colors()
    plot_ly(df, x = ~reorder(node_type, -count), y = ~count,
            type = "bar",
            marker = list(color = colors[df$node_type]),
            text = ~paste0(node_type, ": ", format(count, big.mark = ",")),
            hoverinfo = "text") %>%
      layout(
        xaxis = list(title = "", tickangle = -45),
        yaxis = list(title = "Count", type = "log"),
        margin = list(b = 100),
        showlegend = FALSE
      )
  })

  output$overview_edge_plot <- renderPlotly({
    df <- edge_relation_summary
    plot_ly(df, x = ~reorder(relation, -count), y = ~count,
            type = "bar",
            marker = list(color = "#2b6a99"),
            text = ~paste0(relation, ": ", format(count, big.mark = ",")),
            hoverinfo = "text") %>%
      layout(
        xaxis = list(title = "", tickangle = -45),
        yaxis = list(title = "Count", type = "log"),
        margin = list(b = 120),
        showlegend = FALSE
      )
  })

  # ── Module servers ──
  kg_explorer_server("kg_explorer")
  glycan_predictor_server("glycan_predictor")
  glycan_retrieval_server("glycan_retrieval")
  celltype_server("celltype")
  benchmark_server("benchmark")
  embedding_server("embedding")
  hierarchy_server("hierarchy")
}

# ── Launch ──
shinyApp(ui = ui, server = server)
