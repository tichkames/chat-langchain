export type Source = {
  url: string;
  title: string;
};

export type Message = {
  id: string;
  createdAt?: Date;
  content: string;
  type: "system" | "human" | "ai" | "function";
  sources?: Source[];
  name?: string;
  function_call?: { name: string };
};

export type Feedback = {
  feedback_id: string;
  score: number;
  comment?: string;
};

export type ModelOptions =
  | "openai/gpt-4o-mini"
  | "anthropic/claude-3-5-haiku-20241022"
  | "groq/llama3-70b-8192"
  | "google_genai/gemini-pro";

  export interface StreamToolCall {
    id: string;
    name: string;
    args: string;
    result?: any;
  }
  
  export interface StreamMessage {
    id: string;
    type: 'human' | 'ai' | 'tool';
    content: string;
    toolContent?: StreamToolCall;
  }
  
  export interface StreamEvent {
    event: string;
    name: string;
    run_id: string;
    data: {
      run_id?: string;
      input?: any;
      output?: any;
      chunk?: {
        id?: string;
        content?: any;
        additional_kwargs?: Record<string, any>;
      };
    };
  }
  
  export interface StreamStatus {
    content: string;
    name?: string;
  }