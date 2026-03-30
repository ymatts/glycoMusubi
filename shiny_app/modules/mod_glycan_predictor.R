# glycoMusubi Shiny App — Glycan Predictor Module
# N-linked glycosylation site prediction and protein glycan lookup.

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
glycan_predictor_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(
    sidebar = sidebar(
      title = "Query",
      width = 300,
      textInput(ns("uniprot_id"), "UniProt ID",
                placeholder = "e.g. P12345"),
      actionButton(ns("predict"), "Predict",
                   class = "btn-primary w-100",
                   icon = icon("search-plus"))
    ),

    navset_card_tab(
      id = ns("result_tabs"),

      # ── Tab 1: Protein Info ──
      nav_panel(
        title = "Protein Info",
        icon = icon("id-card"),
        uiOutput(ns("protein_info_card")),
        card(
          card_header("Known Glycosylation Sites (from KG)"),
          card_body(DTOutput(ns("known_sites_table")))
        )
      ),

      # ── Tab 2: Site Predictions ──
      nav_panel(
        title = "Site Predictions",
        icon = icon("bullseye"),
        uiOutput(ns("predictions_ui"))
      ),

      # ── Tab 3: Known Glycans ──
      nav_panel(
        title = "Known Glycans",
        icon = icon("cubes"),
        card(
          card_header("Associated Glycans"),
          card_body(DTOutput(ns("known_glycans_table")))
        )
      )
    )
  )
}

# ══════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════
glycan_predictor_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    # ── Reactive: protein record ──
    protein <- reactiveVal(NULL)
    protein_edges <- reactiveVal(data.frame())
    protein_preds <- reactiveVal(data.frame())

    observeEvent(input$predict, {
      uid <- trimws(input$uniprot_id)
      req(nchar(uid) > 0)

      # Look up protein in proteins_df
      match <- proteins_df %>%
        filter(if ("uniprot_id" %in% names(.)) uniprot_id == uid
               else if ("node_id" %in% names(.)) node_id == uid
               else FALSE)

      if (nrow(match) == 0) {
        protein(NULL)
        protein_edges(data.frame())
        protein_preds(data.frame())
        showNotification(paste("Protein", uid, "not found in the knowledge graph."),
                         type = "warning")
        return()
      }

      protein(match)

      # Filter KG edges for this protein (glycosylation-related)
      edges <- kg_edges %>%
        filter(source_id == uid | target_id == uid) %>%
        filter(relation %in% c("has_glycan", "glycosylated_at", "has_site"))
      protein_edges(edges)

      # Filter site predictions
      if (nrow(site_predictions) > 0 && "uniprot_id" %in% names(site_predictions)) {
        preds <- site_predictions %>% filter(uniprot_id == uid)
        protein_preds(preds)
      } else {
        protein_preds(data.frame())
      }
    })

    # ── Protein info card ──
    output$protein_info_card <- renderUI({
      prot <- protein()
      if (is.null(prot)) {
        return(card(
          card_body(
            p(class = "text-muted",
              "Enter a UniProt ID and click Predict to view protein information.")
          )
        ))
      }

      gene_name <- if ("gene_name" %in% names(prot)) prot$gene_name[1] else "N/A"
      organism  <- if ("organism" %in% names(prot)) prot$organism[1] else "N/A"
      uid <- if ("uniprot_id" %in% names(prot)) prot$uniprot_id[1] else prot$node_id[1]

      # Count glycosylation sites from edges
      glyco_edges <- protein_edges()
      n_sites <- glyco_edges %>%
        filter(relation %in% c("glycosylated_at", "has_site")) %>%
        nrow()

      card(
        card_header("Protein Metadata"),
        card_body(
          layout_columns(
            col_widths = c(3, 3, 3, 3),
            fill = FALSE,
            value_box_ui(uid, "UniProt ID", "fingerprint"),
            value_box_ui(gene_name, "Gene Name", "tag"),
            value_box_ui(organism, "Organism", "globe"),
            value_box_ui(n_sites, "Glycosylation Sites", "map-pin")
          )
        )
      )
    })

    # ── Known glycosylation sites table ──
    output$known_sites_table <- renderDT({
      edges <- protein_edges()
      if (nrow(edges) == 0) {
        return(datatable(data.frame(Message = "No glycosylation site data available."),
                         options = list(dom = "t"), rownames = FALSE))
      }

      site_edges <- edges %>%
        filter(relation %in% c("glycosylated_at", "has_site"))

      if (nrow(site_edges) == 0) {
        return(datatable(data.frame(Message = "No glycosylation sites recorded in KG."),
                         options = list(dom = "t"), rownames = FALSE))
      }

      display_df <- site_edges %>%
        select(any_of(c("source_id", "target_id", "relation"))) %>%
        rename_with(~ str_replace_all(.x, "_", " ") %>% str_to_title())

      datatable(display_df,
                rownames = FALSE,
                options = list(pageLength = 10, scrollX = TRUE),
                class = "compact stripe hover")
    })

    # ── Site predictions UI ──
    output$predictions_ui <- renderUI({
      ns <- session$ns
      prot <- protein()

      if (is.null(prot)) {
        return(card(
          card_body(
            p(class = "text-muted",
              "Enter a UniProt ID and click Predict to view site predictions.")
          )
        ))
      }

      preds <- protein_preds()

      if (nrow(preds) == 0) {
        return(card(
          card_header(icon("info-circle"), " Predictions Not Available"),
          card_body(
            class = "bg-light",
            p("Pre-computed N-linked glycosylation site predictions are not yet",
              "available for this protein."),
            p("The prediction pipeline produces results for proteins with",
              "sufficient sequence context in the training set. Future updates",
              "may include additional predictions."),
            tags$hr(),
            tags$small(class = "text-muted",
              "See the Benchmarks tab for overall prediction performance metrics.")
          )
        ))
      }

      card(
        card_header("Predicted N-Linked Glycosylation Sites"),
        card_body(DTOutput(ns("predictions_table")))
      )
    })

    # ── Site predictions table ──
    output$predictions_table <- renderDT({
      preds <- protein_preds()
      req(nrow(preds) > 0)

      display_cols <- intersect(
        c("position", "residue", "f1_score", "predicted_type", "cluster_id"),
        names(preds)
      )

      display_df <- preds %>%
        select(all_of(display_cols))

      # Rename for display
      col_names <- c(
        position = "Position",
        residue = "Residue",
        f1_score = "Predicted Probability",
        predicted_type = "Glycosylation Type",
        cluster_id = "Structural Cluster"
      )

      display_df <- display_df %>%
        rename(any_of(col_names))

      datatable(display_df,
                rownames = FALSE,
                options = list(pageLength = 15, scrollX = TRUE,
                               order = list(list(0, "asc"))),
                class = "compact stripe hover") %>%
        formatRound(columns = "Predicted Probability", digits = 3)
    })

    # ── Known glycans table ──
    output$known_glycans_table <- renderDT({
      edges <- protein_edges()
      if (nrow(edges) == 0) {
        return(datatable(data.frame(Message = "No associated glycans found."),
                         options = list(dom = "t"), rownames = FALSE))
      }

      glycan_edges <- edges %>% filter(relation == "has_glycan")

      if (nrow(glycan_edges) == 0) {
        return(datatable(data.frame(Message = "No has_glycan relations found for this protein."),
                         options = list(dom = "t"), rownames = FALSE))
      }

      # Get glycan IDs (target of has_glycan relation)
      glycan_ids <- glycan_edges$target_id

      # Join with glycans_df for details
      if (nrow(glycans_df) > 0) {
        id_col <- if ("glytoucan_id" %in% names(glycans_df)) "glytoucan_id"
                  else if ("node_id" %in% names(glycans_df)) "node_id"
                  else names(glycans_df)[1]

        glycan_info <- glycans_df %>%
          filter(.data[[id_col]] %in% glycan_ids)

        if (nrow(glycan_info) > 0) {
          # Truncate long structural columns
          char_cols <- names(glycan_info)[sapply(glycan_info, is.character)]
          glycan_info <- glycan_info %>%
            mutate(across(all_of(char_cols), ~ truncate_str(.x, 80)))

          return(datatable(glycan_info,
                           rownames = FALSE,
                           options = list(pageLength = 10, scrollX = TRUE),
                           class = "compact stripe hover"))
        }
      }

      # Fallback: just show IDs
      datatable(data.frame(glycan_id = glycan_ids),
                rownames = FALSE,
                options = list(pageLength = 10, scrollX = TRUE),
                class = "compact stripe hover")
    })
  })
}
