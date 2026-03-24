use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use colored::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use unicode_segmentation::UnicodeSegmentation;
use walkdir::WalkDir;
// --- 数据结构 ---

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

#[derive(Serialize, Deserialize, Clone)]
struct VecEntry {
    path: String,
    content: String,
    embedding: Vec<f32>,
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

        // 简单语义切分：比较相邻句子的相似度
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

// --- 业务逻辑 ---

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
                println!("{} does not exist.", file_path.display());
            }
        }
        Commands::Delete { title } => {
            let file_path = kb_dir.join(format!("{}.md", title));
            if file_path.exists() {
                fs::remove_file(file_path)?;
            } else {
                println!("{} does not exist.", file_path.display());
            }
        }
        Commands::Sync => {
            println!("🔄 Processing semantic chunks and embeddings...");
            let mut all_entries = Vec::new();
            for entry in WalkDir::new(&kb_dir).into_iter().filter_map(|e| e.ok()) {
                if entry.path().is_file() && entry.path().extension().map_or(false, |s| s == "md") {
                    let content = fs::read_to_string(entry.path())?;
                    let chunks = SemanticChunker::chunk(&content, &ai).await;
                    for chunk in chunks {
                        let embedding = ai.get_embedding(&chunk).await?;
                        all_entries.push(VecEntry {
                            path: entry.file_name().to_string_lossy().into(),
                            content: chunk,
                            embedding,
                        });
                    }
                }
            }
            let encoded: Vec<u8> = bincode::serialize(&all_entries)?;
            fs::write(kb_dir.join("index.bin"), encoded)?;
            println!("✅ Sync complete. Binary index saved.");
        }
        Commands::Ask { question } => {
            let index_bin = kb_dir.join("index.bin");
            if !index_bin.exists() {
                return Err(anyhow!("Please run 'sync' first."));
            }

            // 1. 获取问题向量
            let query_emb = ai.get_embedding(question).await?;

            // 2. 加载二进制索引
            let bin_data = fs::read(index_bin)?;
            let data: Vec<VecEntry> = bincode::deserialize(&bin_data)?;

            // 3. 性能优化：先计算相似度，再排序
            // 使用 rayon 并行计算所有分块的相似度，并存储为 (相似度, 索引) 的元组
            let mut scores: Vec<(f32, &VecEntry)> = data
                .par_iter() // 并行计算
                .map(|entry| {
                    let sim = cosine_similarity(&query_emb, &entry.embedding);
                    (sim, entry)
                })
                .collect();

            // 4. 对分值进行排序（从高到低）
            // 注意：f32 没有实现 Ord，所以用 partial_cmp
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // 5. 取前 3 个最相关的片段作为上下文
            let context = scores
                .iter()
                .take(3)
                .map(|(_sim, entry)| entry.content.as_str())
                .collect::<Vec<_>>()
                .join("\n---\n");

            // 6. 构造 Prompt 并询问 AI
            let prompt = format!(
                "Context:\n{}\n\nQuestion: {}\nAnswer based only on context:",
                context, question
            );

            println!("🤖 AI is thinking...");
            let answer = ai.chat(&prompt).await?;
            println!("\n{}\n{}", "Answer:".green().bold(), answer);
        }
    }
    Ok(())
}
