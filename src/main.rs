use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use colored::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::SystemTime;
use unicode_segmentation::UnicodeSegmentation;
use walkdir::WalkDir;

// --- 数据结构 ---

#[derive(Serialize, Deserialize)]
struct FullIndex {
    // 存储 文件名 -> 最后修改时间
    metadata: HashMap<String, SystemTime>,
    // 所有的向量分块
    entries: Vec<VecEntry>,
}

#[derive(Serialize, Deserialize, Clone)]
struct VecEntry {
    path: String,
    content: String,
    embedding: Vec<f32>,
}

#[derive(Parser)]
#[command(name = "mkb", about = "Rust Minimalist Knowledge Base with RAG", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 创建新笔记
    New { title: String },
    /// 列出所有笔记
    List,
    /// 编辑笔记
    Edit { title: String },
    /// 删除笔记
    Delete { title: String },
    /// 同步并建立语义索引 (RAG 准备)
    Sync,
    /// 基于你的知识库提问 (RAG)
    Ask { question: String },
}

// --- AI 客户端 (OpenAI/Ollama 兼容) ---

struct AiClient {
    api_key: String,
    base_url: String,
    model_emb: String,
    model_chat: String,
}

impl AiClient {
    fn from_env() -> Self {
        Self {
            api_key: std::env::var("AI_API_KEY").unwrap_or_else(|_| "ollama".into()),
            base_url: std::env::var("AI_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".into()),
            model_emb: std::env::var("AI_EMB_MODEL").unwrap_or_else(|_| "nomic-embed-text".into()),
            model_chat: std::env::var("AI_CHAT_MODEL").unwrap_or_else(|_| "llama3".into()),
        }
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let client = reqwest::Client::new();
        let res: serde_json::Value = client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({ "input": text, "model": &self.model_emb }))
            .send()
            .await?
            .json()
            .await?;

        let vec = res["data"][0]["embedding"]
            .as_array()
            .ok_or(anyhow!("Failed to get embedding: {:?}", res))?
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        Ok(vec)
    }

    async fn chat(&self, prompt: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let res: serde_json::Value = client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": &self.model_chat,
                "messages": [{"role": "user", "content": prompt}]
            }))
            .send()
            .await?
            .json()
            .await?;

        Ok(res["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No response")
            .to_string())
    }
}

// --- 语义工具 ---

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

struct SemanticChunker;
impl SemanticChunker {
    async fn chunk(text: &str, ai: &AiClient) -> Vec<String> {
        let sentences: Vec<&str> = text.unicode_sentences().filter(|s| s.len() > 5).collect();
        if sentences.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = sentences[0].to_string();

        for i in 0..sentences.len() - 1 {
            let e1 = ai.get_embedding(sentences[i]).await.unwrap_or_default();
            let e2 = ai.get_embedding(sentences[i + 1]).await.unwrap_or_default();

            if !e1.is_empty() && !e2.is_empty() && cosine_similarity(&e1, &e2) > 0.7 {
                current_chunk.push_str(" ");
                current_chunk.push_str(sentences[i + 1]);
            } else {
                chunks.push(current_chunk);
                current_chunk = sentences[i + 1].to_string();
            }
        }
        chunks.push(current_chunk);
        chunks
    }
}

fn get_kb_dir() -> PathBuf {
    let mut path = dirs::home_dir().expect("Could not find home dir");
    path.push(".mkb");
    if !path.exists() {
        fs::create_dir_all(&path).unwrap();
    }
    path
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let kb_dir = get_kb_dir();
    let ai = AiClient::from_env();

    match &cli.command {
        Commands::New { title } => {
            let file_path = kb_dir.join(format!("{}.md", title));
            if !file_path.exists() {
                fs::write(&file_path, format!("# {}\n\n", title))?;
            }
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
            Command::new(editor).arg(file_path).status()?;
        }
        Commands::List => {
            println!("{}", "Your Knowledge Base:".bold().blue());
            for entry in WalkDir::new(&kb_dir).into_iter().filter_map(|e| e.ok()) {
                if entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
                    println!(" - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        Commands::Edit { title } => {
            let file_path = kb_dir.join(format!("{}.md", title));
            if file_path.exists() {
                let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
                Command::new(editor).arg(file_path).status()?;
            } else {
                println!("{} does not exist.", file_path.display().to_string().red());
            }
        }
        Commands::Delete { title } => {
            let file_path = kb_dir.join(format!("{}.md", title));
            if file_path.exists() {
                fs::remove_file(file_path)?;
                println!("Deleted {}", title.yellow());
            } else {
                println!("{} does not exist.", title.red());
            }
        }
        Commands::Sync => {
            let index_bin = kb_dir.join("index.bin");

            // 1. 加载旧索引
            let full_index = if index_bin.exists() {
                let bin_data = fs::read(&index_bin)?;
                bincode::deserialize::<FullIndex>(&bin_data).unwrap_or(FullIndex {
                    metadata: HashMap::new(),
                    entries: Vec::new(),
                })
            } else {
                FullIndex {
                    metadata: HashMap::new(),
                    entries: Vec::new(),
                }
            };

            println!("🔄 Checking for changes...");

            let mut new_entries = Vec::new();
            let mut updated_metadata = HashMap::new();

            // 2. 遍历磁盘文件
            for entry in WalkDir::new(&kb_dir).into_iter().filter_map(|e| e.ok()) {
                if entry.path().is_file() && entry.path().extension().map_or(false, |s| s == "md") {
                    let path_str = entry.file_name().to_string_lossy().to_string();
                    let mtime = entry.metadata()?.modified()?;

                    // 增量检查
                    if let Some(old_mtime) = full_index.metadata.get(&path_str) {
                        if *old_mtime == mtime {
                            println!("  - {} (unchanged)", path_str.dimmed());
                            let old_chunks: Vec<VecEntry> = full_index.entries.iter()
                                .filter(|e| e.path == path_str)
                                .cloned()
                                .collect();
                            new_entries.extend(old_chunks);
                            updated_metadata.insert(path_str, mtime);
                            continue;
                        }
                    }

                    // 更新或新增
                    println!("  - {} (updating...)", path_str.yellow().bold());
                    let content = fs::read_to_string(entry.path())?;
                    let chunks = SemanticChunker::chunk(&content, &ai).await;
                    for chunk in chunks {
                        let embedding = ai.get_embedding(&chunk).await?;
                        new_entries.push(VecEntry {
                            path: path_str.clone(),
                            content: chunk, // 这里修复了之前的 chunk_content 错误
                            embedding,
                        });
                    }
                    updated_metadata.insert(path_str, mtime);
                }
            }

            // 3. 保存新索引
            let new_full_index = FullIndex {
                metadata: updated_metadata,
                entries: new_entries,
            };

            let encoded = bincode::serialize(&new_full_index)?;
            fs::write(index_bin, encoded)?;
            println!("✅ Sync complete. Total chunks: {}", new_full_index.entries.len());
        }
        Commands::Ask { question } => {
            let index_bin = kb_dir.join("index.bin");
            if !index_bin.exists() {
                return Err(anyhow!("Please run 'sync' first."));
            }

            let query_emb = ai.get_embedding(question).await?;
            let bin_data = fs::read(index_bin)?;
            let full_index: FullIndex = bincode::deserialize(&bin_data)?;
            
            // 性能优化：并行计算相似度
            let mut scores: Vec<(f32, &VecEntry)> = full_index.entries
                .par_iter()
                .map(|entry| {
                    let sim = cosine_similarity(&query_emb, &entry.embedding);
                    (sim, entry)
                })
                .collect();

            // 排序并取 Top 3
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let context = scores.iter().take(3)
                .map(|(_sim, entry)| entry.content.as_str())
                .collect::<Vec<_>>()
                .join("\n---\n");

            let prompt = format!(
                "Use the context to answer the question. Context:\n{}\n\nQuestion: {}",
                context, question
            );

            println!("🤖 AI is thinking...");
            let answer = ai.chat(&prompt).await?;
            println!("\n{}\n{}", "Answer:".green().bold(), answer);
        }
    }
    Ok(())
}