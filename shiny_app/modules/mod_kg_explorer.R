# glycoMusubi Shiny App — KG Explorer Module
# Search and visualize subgraphs of the knowledge graph.

# ── UI ──
kg_explorer_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(
    sidebar = sidebar(
      title = "Search KG",
      width = 300,
      textInput(ns("search_query"), "Search Query",
                placeholder = "e.g. P12345, G00055MO"),
      selectInput(ns("node_type_filter"), "Node Type",
                  choices = c("All", sort(unique(kg_nodes$node_type)))),
      numericInput(ns("max_neighbors"), "Max Neighbors",
                   value = 50, min = 1, max = 500, step = 10),
      actionButton(ns("btn_search"), "Search",
                   class = "btn-primary w-100",
                   icon = icon("search"))
    ),

    # ── Value boxes ──
    layout_columns(
      fill = FALSE,
      col_widths = c(4, 4, 4),
      uiOutput(ns("vb_matched_nodes")),
      uiOutput(ns("vb_connected_edges")),
      uiOutput(ns("vb_neighbor_types"))
    ),

    # ── Network graph ──
    card(
      card_header("Subgraph Visualization"),
      card_body(visNetworkOutput(ns("vis_subgraph"), height = "500px"))
    ),

    # ── Detail tables ──
    layout_columns(
      col_widths = c(6, 6),
      card(
        card_header("Matched Nodes"),
        card_body(DTOutput(ns("dt_nodes")))
      ),
      card(
        card_header("Connected Edges"),
        card_body(DTOutput(ns("dt_edges")))
      )
    )
  )
}

# ── Server ──
kg_explorer_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    # Reactive: search results
    search_results <- eventReactive(input$btn_search, {
      req(nchar(trimws(input$search_query)) > 0)

      query <- trimws(input$search_query)
      max_n <- input$max_neighbors

      # Filter nodes by search query (case-insensitive grep on node_id)
      matched <- kg_nodes[grepl(query, kg_nodes$node_id, ignore.case = TRUE), ]

      # Optionally filter by node type
      if (input$node_type_filter != "All") {
        matched <- matched[matched$node_type == input$node_type_filter, ]
      }

      if (nrow(matched) == 0) {
        return(list(
          matched_nodes = matched,
          edges = data.frame(),
          all_node_ids = character(0),
          neighbor_types = character(0)
        ))
      }

      matched_ids <- matched$node_id

      # Find edges connected to matched nodes
      edges <- kg_edges[
        kg_edges$source_id %in% matched_ids | kg_edges$target_id %in% matched_ids,
      ]

      # Limit edge count
      if (nrow(edges) > max_n) {
        edges <- edges[seq_len(max_n), ]
      }

      # Collect all node IDs appearing in the subgraph
      all_node_ids <- unique(c(matched_ids, edges$source_id, edges$target_id))

      # Determine unique neighbor types (nodes not in matched set)
      neighbor_ids <- setdiff(all_node_ids, matched_ids)
      neighbor_rows <- kg_nodes[kg_nodes$node_id %in% neighbor_ids, ]
      neighbor_types <- unique(neighbor_rows$node_type)

      list(
        matched_nodes = matched,
        edges = edges,
        all_node_ids = all_node_ids,
        neighbor_types = neighbor_types
      )
    })

    # Value boxes
    output$vb_matched_nodes <- renderUI({
      res <- search_results()
      value_box_ui(nrow(res$matched_nodes), "Matched Nodes",
                   icon = "bullseye", color = "#2b6a99")
    })

    output$vb_connected_edges <- renderUI({
      res <- search_results()
      value_box_ui(nrow(res$edges), "Connected Edges",
                   icon = "share-alt", color = "#3a7db5")
    })

    output$vb_neighbor_types <- renderUI({
      res <- search_results()
      value_box_ui(length(res$neighbor_types), "Unique Neighbor Types",
                   icon = "layer-group", color = "#4a9fd4")
    })

    # Network visualization
    output$vis_subgraph <- renderVisNetwork({
      res <- search_results()

      if (length(res$all_node_ids) == 0) {
        visNetwork(data.frame(), data.frame()) %>%
          visOptions(nodesIdSelection = FALSE) %>%
          visLayout(randomSeed = 42)
      } else {
        vis_nodes <- build_vis_nodes(res$all_node_ids, kg_nodes)
        vis_edges <- build_vis_edges(res$edges)

        visNetwork(vis_nodes, vis_edges) %>%
          visOptions(
            highlightNearest = list(enabled = TRUE, degree = 1),
            nodesIdSelection = TRUE,
            selectedBy = "group"
          ) %>%
          visPhysics(
            solver = "forceAtlas2Based",
            forceAtlas2Based = list(gravitationalConstant = -50)
          ) %>%
          visLayout(randomSeed = 42) %>%
          visInteraction(navigationButtons = TRUE)
      }
    })

    # Node details table
    output$dt_nodes <- DT::renderDT({
      res <- search_results()
      df <- kg_nodes[kg_nodes$node_id %in% res$all_node_ids, ]
      if ("node_id" %in% names(df)) {
        df$node_id <- truncate_str(df$node_id, 50)
      }
      datatable(df, rownames = FALSE, filter = "top",
                options = list(pageLength = 10, scrollX = TRUE))
    })

    # Edge details table
    output$dt_edges <- DT::renderDT({
      res <- search_results()
      df <- res$edges
      if (nrow(df) > 0) {
        df$source_id <- truncate_str(df$source_id, 40)
        df$target_id <- truncate_str(df$target_id, 40)
      }
      datatable(df, rownames = FALSE, filter = "top",
                options = list(pageLength = 10, scrollX = TRUE))
    })
  })
}
