//! Status view - show model, memory, and session stats

use eframe::egui::{Color32, ProgressBar, RichText, Ui};

use crate::desktop::state::{UiMessage, UiState};

/// Warning amber for external-provider indicators.
const EGRESS_WARN_COLOR: Color32 = Color32::from_rgb(251, 191, 36); // #fbbf24
/// Safe green for local-only indicators.
const EGRESS_SAFE_COLOR: Color32 = Color32::from_rgb(74, 222, 128); // #4ade80

pub struct StatusView;

impl StatusView {
    pub fn show(ui: &mut Ui, state: &mut UiState) -> Option<UiMessage> {
        let mut message_to_send = None;

        ui.heading("Status");
        ui.add_space(10.0);

        // Refresh button
        if ui.button("Refresh").clicked() {
            message_to_send = Some(UiMessage::RefreshStatus);
        }

        ui.add_space(10.0);

        // Model info
        ui.group(|ui| {
            ui.label(RichText::new("Model").strong());
            ui.label(&state.model);
        });

        ui.add_space(10.0);

        // Data egress info
        ui.group(|ui| {
            ui.label(RichText::new("Data Egress").strong());
            if let Some(ref egress) = state.egress {
                if egress.external_llm || egress.external_embeddings {
                    ui.label(
                        RichText::new("\u{26A0} External providers active")
                            .color(EGRESS_WARN_COLOR),
                    );
                    ui.label(
                        RichText::new(&egress.summary)
                            .small()
                            .color(EGRESS_WARN_COLOR),
                    );
                    for detail in &egress.details {
                        ui.label(RichText::new(detail).small().color(Color32::GRAY));
                    }
                } else {
                    ui.label(
                        RichText::new("\u{2713} Local only — no data leaves this device")
                            .color(EGRESS_SAFE_COLOR),
                    );
                }
            } else {
                ui.label(RichText::new("Loading...").color(Color32::GRAY));
            }
        });

        ui.add_space(10.0);

        // Memory info
        ui.group(|ui| {
            ui.label(RichText::new("Memory").strong());
            ui.label(format!("Chunks: {}", state.memory_chunks));
            ui.horizontal(|ui| {
                ui.label("Embeddings:");
                if state.has_embeddings {
                    ui.label(RichText::new("enabled").color(Color32::from_rgb(46, 204, 113)));
                } else {
                    ui.label(RichText::new("disabled").color(Color32::GRAY));
                }
            });
        });

        ui.add_space(10.0);

        // Session info
        if let Some(ref status) = state.status {
            ui.group(|ui| {
                ui.label(RichText::new("Session").strong());
                ui.label(format!("ID: {}...", &status.id[..8.min(status.id.len())]));
                ui.label(format!("Messages: {}", status.message_count));
                ui.label(format!("Compactions: {}", status.compaction_count));

                // Token usage bar
                ui.add_space(5.0);
                ui.label("Context usage:");
                let token_pct = status.token_count as f32 / 128000.0; // Assume 128k context
                ui.add(
                    ProgressBar::new(token_pct.min(1.0))
                        .text(format!("~{} tokens", status.token_count)),
                );

                if token_pct > 0.8 {
                    ui.label(
                        RichText::new("Context nearly full. Consider starting a new session.")
                            .color(Color32::from_rgb(231, 76, 60))
                            .small(),
                    );
                }
            });

            ui.add_space(10.0);

            // API usage
            if status.api_input_tokens > 0 || status.api_output_tokens > 0 {
                ui.group(|ui| {
                    ui.label(RichText::new("API Usage (Session)").strong());
                    ui.label(format!("Input: {} tokens", status.api_input_tokens));
                    ui.label(format!("Output: {} tokens", status.api_output_tokens));
                    ui.label(format!(
                        "Total: {} tokens",
                        status.api_input_tokens + status.api_output_tokens
                    ));
                });
            }
        }

        message_to_send
    }
}
