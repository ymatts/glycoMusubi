# glycoMusubi Shiny App — Utility Functions

#' Create a value box for metric display
value_box_ui <- function(value, label, icon = NULL, color = "#2b6a99") {
  div(
    class = "value-box",
    style = paste0("border-left: 4px solid ", color, ";"),
    if (!is.null(icon)) icon(icon, class = "fa-2x", style = paste0("color:", color)),
    div(class = "value", format(value, big.mark = ",")),
    div(class = "label", label)
  )
}

#' Format numeric metric for display
fmt_metric <- function(x, digits = 3) {
  if (is.na(x)) return("—")
  formatC(x, format = "f", digits = digits)
}

#' Truncate long strings for table display
truncate_str <- function(x, max_len = 60) {
  ifelse(nchar(x) > max_len, paste0(substr(x, 1, max_len - 3), "..."), x)
}

#' Map node types to display colors
node_type_colors <- function() {
  c(
    glycan = "#2b6a99",
    protein = "#3a7db5",
    enzyme = "#4a9fd4",
    site = "#6b97b5",
    disease = "#c04e3f",
    compound = "#d4764a",
    variant = "#8b6b99",
    motif = "#5a9977",
    reaction = "#99855a",
    cellular_location = "#7a8a99"
  )
}

#' Map node types to icons
node_type_icons <- function() {
  c(
    glycan = "cubes",
    protein = "dna",
    enzyme = "flask",
    site = "map-pin",
    disease = "heartbeat",
    compound = "pills",
    variant = "random",
    motif = "puzzle-piece",
    reaction = "exchange-alt",
    cellular_location = "building"
  )
}

#' Safe read of TSV with fallback
safe_read_tsv <- function(path, ...) {
  if (!file.exists(path)) {
    warning(paste("File not found:", path))
    return(data.frame())
  }
  readr::read_tsv(path, show_col_types = FALSE, ...)
}

#' Safe read of JSON
safe_read_json <- function(path) {
  if (!file.exists(path)) return(list())
  jsonlite::fromJSON(path)
}

#' Build visNetwork edge data from KG edges
build_vis_edges <- function(edges_df, max_edges = 500) {
  if (nrow(edges_df) == 0) return(data.frame())
  df <- head(edges_df, max_edges)
  data.frame(
    from = df$source_id,
    to = df$target_id,
    label = df$relation,
    arrows = "to",
    color = "#b0b5bb",
    smooth = TRUE,
    stringsAsFactors = FALSE
  )
}

#' Build visNetwork node data
build_vis_nodes <- function(node_ids, nodes_df) {
  colors <- node_type_colors()
  matched <- nodes_df[nodes_df$node_id %in% node_ids, ]
  if (nrow(matched) == 0) return(data.frame())
  data.frame(
    id = matched$node_id,
    label = ifelse(nchar(matched$node_id) > 20,
                   paste0(substr(matched$node_id, 1, 17), "..."),
                   matched$node_id),
    group = matched$node_type,
    title = paste0("<b>", matched$node_id, "</b><br>Type: ", matched$node_type),
    color = colors[matched$node_type],
    stringsAsFactors = FALSE
  )
}
