# glycoMusubi Shiny App — Hierarchy Browser Module
# Explore the glycan structural taxonomy via BFS tree traversal.

# ── UI ──
hierarchy_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(
    sidebar = sidebar(
      title = "Hierarchy Explorer",
      width = 300,
      textInput(ns("root_id"), "Root Glycan ID",
                placeholder = "e.g. G00055MO"),
      selectInput(ns("relation_type"), "Relation Type",
                  choices = c("parent_of", "child_of")),
      numericInput(ns("max_depth"), "Max Depth",
                   value = 3, min = 1, max = 5, step = 1),
      actionButton(ns("btn_explore"), "Explore",
                   class = "btn-primary w-100",
                   icon = icon("sitemap"))
    ),

    # ── Tree visualization ──
    card(
      card_header("Hierarchy Tree"),
      card_body(visNetworkOutput(ns("vis_tree"), height = "600px"))
    ),

    # ── Subtree node listing ──
    card(
      card_header("Subtree Nodes"),
      card_body(DTOutput(ns("dt_subtree")))
    )
  )
}

# ── Server ──
hierarchy_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    # BFS traversal from root node
    bfs_result <- eventReactive(input$btn_explore, {
      req(nchar(trimws(input$root_id)) > 0)

      root <- trimws(input$root_id)
      rel <- input$relation_type
      max_d <- input$max_depth

      # Use glycan_hierarchy if available; fall back to kg_edges
      hier <- if (nrow(glycan_hierarchy) > 0) {
        glycan_hierarchy[glycan_hierarchy$relation == rel, ]
      } else {
        kg_edges[kg_edges$relation == rel, ]
      }

      # BFS
      visited <- list()
      visited[[root]] <- 0L
      queue <- list(list(node = root, depth = 0L))
      tree_edges <- data.frame(
        from = character(0), to = character(0),
        stringsAsFactors = FALSE
      )

      while (length(queue) > 0) {
        current <- queue[[1]]
        queue <- queue[-1]

        if (current$depth >= max_d) next

        # Find children: source_id -> target_id for the given relation
        children_rows <- hier[hier$source_id == current$node, ]
        child_ids <- children_rows$target_id

        for (cid in child_ids) {
          if (is.null(visited[[cid]])) {
            new_depth <- current$depth + 1L
            visited[[cid]] <- new_depth
            queue <- c(queue, list(list(node = cid, depth = new_depth)))
            tree_edges <- rbind(tree_edges,
                                data.frame(from = current$node, to = cid,
                                           stringsAsFactors = FALSE))
          }
        }
      }

      # Build node data frame with depth
      node_ids <- names(visited)
      depths <- as.integer(unlist(visited))

      nodes_df <- data.frame(
        node_id = node_ids,
        depth = depths,
        stringsAsFactors = FALSE
      )

      # Merge with kg_nodes for type info where available
      if (nrow(kg_nodes) > 0) {
        extra <- kg_nodes[kg_nodes$node_id %in% node_ids,
                          intersect(names(kg_nodes), c("node_id", "node_type"))]
        nodes_df <- merge(nodes_df, extra, by = "node_id", all.x = TRUE)
      }

      list(nodes = nodes_df, edges = tree_edges)
    })

    # Tree visualization
    output$vis_tree <- renderVisNetwork({
      res <- bfs_result()

      if (nrow(res$nodes) == 0) {
        visNetwork(data.frame(), data.frame()) %>%
          visLayout(randomSeed = 42)
      } else {
        # Color palette by depth level
        depth_colors <- c("#2b6a99", "#3a7db5", "#4a9fd4",
                          "#6b97b5", "#8aafcc", "#a8c7dd")
        max_d <- max(res$nodes$depth, na.rm = TRUE)

        vis_nodes <- data.frame(
          id = res$nodes$node_id,
          label = ifelse(nchar(res$nodes$node_id) > 20,
                         paste0(substr(res$nodes$node_id, 1, 17), "..."),
                         res$nodes$node_id),
          level = res$nodes$depth,
          color = depth_colors[pmin(res$nodes$depth + 1L, length(depth_colors))],
          title = paste0(
            "<b>", res$nodes$node_id, "</b>",
            "<br>Depth: ", res$nodes$depth,
            if ("node_type" %in% names(res$nodes))
              paste0("<br>Type: ", res$nodes$node_type) else ""
          ),
          stringsAsFactors = FALSE
        )

        vis_edges <- data.frame(
          from = res$edges$from,
          to = res$edges$to,
          arrows = "to",
          color = "#b0b5bb",
          stringsAsFactors = FALSE
        )

        visNetwork(vis_nodes, vis_edges) %>%
          visHierarchicalLayout(direction = "UD", sortMethod = "directed") %>%
          visOptions(
            highlightNearest = list(enabled = TRUE, degree = 1),
            nodesIdSelection = TRUE
          ) %>%
          visInteraction(navigationButtons = TRUE) %>%
          visLayout(randomSeed = 42)
      }
    })

    # Subtree node table
    output$dt_subtree <- DT::renderDT({
      res <- bfs_result()
      df <- res$nodes[order(res$nodes$depth, res$nodes$node_id), ]

      # Add depth-level summary as a footer-like row count
      datatable(df, rownames = FALSE, filter = "top",
                options = list(pageLength = 15, scrollX = TRUE),
                caption = paste0(
                  "Total: ", nrow(df), " nodes across ",
                  length(unique(df$depth)), " depth levels"
                ))
    })
  })
}
