import type {
  StreamEvent,
  StreamToolCall,
  StreamStatus,
} from '../types';

interface EventProcessorProps {
  onToken: (token: string) => void;
  onStatus: (status: StreamStatus) => void;
  onComplete: () => void;
  onRecommendations: (name: string, recommendations: any) => void;
}

// Processes events from the stream and updates the UI accordingly.
export const eventProcessor = ({
  onToken,
  onStatus,
  onComplete,
  onRecommendations,
}: EventProcessorProps) => {
  // Process events from the stream, handling each event type appropriately.
  const processEvents = async () => {
    const eventHandlers: Record<string, (event: StreamEvent) => void> = {
      metadata: handleStart,
      end: handleEnd,
      on_tool_start: handleToolStart,
      on_tool_end: handleToolEnd,
      on_chain_start: handleChainStart,
      on_chain_end: handleChainEnd,
      on_chat_model_stream: handleChatModelStream,
      // on_chain_stream: handleChainStream,
    };
  };

  // Handle metadata events.
  const handleStart = (event: StreamEvent) => {
    console.log('handleStart', event);

    // console.log('event run_id', event.run_id);
    // console.log('data run_id', data?.run_id);
  };

  // Handle chat model stream events.
  const handleChatModelStream = (event: StreamEvent) => {
    console.log('handleChatModelStream', event);

    const { data } = event;
    if (data?.chunk) {
      const { content } = data.chunk;

      // if content is an array, as seen in Anthropic responses
      if (Array.isArray(content)) {
        if (content[0]?.text) {
          const contentChunk = content[0]?.text;
          onToken(contentChunk);
        }
      } else {
        if (content && content?.trim().length > 0) {
          onToken(content);
        }
      }
    }
  };

  // Handle the start of a tool or retriever execution.
  const handleToolStart = (event: StreamEvent) => {
    console.log('handleToolStart', event);

    const { data } = event;

    const msg = `Looking up: ${data?.input?.query}`;

    // const msg = `;
    // \n**Calling tool:** \`${
    //   event.name
    // }\` with **args:** \n\`\`\`\n${JSON.stringify(
    //   data?.input,
    //   null,
    //   2
    // )}\n\`\`\`\n
    // `;

    onStatus({
      content: msg,
    });
  };

  // Handle the end of a tool execution.
  const handleToolEnd = (event: StreamEvent) => {
    console.log('handleToolEnd', event);

    const { data } = event;
    const toolCall: StreamToolCall = {
      id: data?.output?.tool_call_id,
      name: data?.output?.name,
      args: data?.input,
      result: data?.output,
    };

    const msg = `Products found`;

    // const msg = `
    //   \n**Ending tool:** \`${
    //     toolCall.name
    //   }\` with **args:** \n\`\`\`\n${JSON.stringify(
    //   toolCall?.args,
    //   null,
    //   2
    // )}\n\`\`\`\n
    // `;

    onStatus({
      content: msg,
    });
  };

  // Handle the end of retriever execution.
  // const handleRetrieverStart = (event: StreamEvent) => {
  //   console.log('handleRetrieverStart', event);

  //   let toolName = event.name;

  //   if (!toolName) {
  //     console.log('no tool name, using event.event');
  //     toolName = event.event;
  //   }

  //   const { data } = event;
  //   const toolCall: StreamToolCall = {
  //     id: toolName,
  //     name: toolName,
  //     args: data?.input,
  //     result: data?.output,
  //   };

  //   const msg = `
  //     \n**Starting Retriever:** \`${toolCall.name}\` with **args:**
  //     \n\`\`\`\n${JSON.stringify(toolCall?.args, null, 2)}\n\`\`\`\n
  //   `;

  //   // onStatus(msg);
  // };

  // Handle the end of retriever execution.
  // const handleRetrieverEnd = (event: StreamEvent) => {
  //   console.log('handleRetrieverEnd', event);

  //   let toolName = event.name;

  //   if (!toolName) {
  //     console.log('no tool name, using event.event');
  //     toolName = event.event;
  //   }

  //   const { data } = event;
  //   const toolCall: StreamToolCall = {
  //     id: toolName,
  //     name: toolName,
  //     args: data?.input,
  //     result: data?.output,
  //   };

  //   const msg = `
  //     \n**Ending Retriever:** \`${toolCall.name}\` with **args:**
  //     \n\`\`\`\n${JSON.stringify(toolCall?.args, null, 2)}\n\`\`\`\n
  //   `;

  //   // onStatus(msg);
  // };

  // Handle the start of chain execution.
  const handleChainStart = (event: StreamEvent) => {
    console.log('handleChainStart', event);

    const { name } = event;

    if (name === 'generate') {
      const msg = `Finalizing your recommendations`;

      onStatus({
        content: msg,
      });
    }
  };

  // Handle the end of chain execution.
  const handleChainEnd = (event: StreamEvent) => {
    console.log('handleChainEnd', event);

    const { name } = event;

    if (name === 'generate') {
      const data = event.data;
      if (data?.output) {
        if (data?.output?.product_recommendations) {
          onRecommendations('product', data?.output?.product_recommendations);
        }

        if (data?.output?.menu_recommendations) {
          onRecommendations('menu', data?.output?.menu_recommendations);
        }
      }
    }
  };

  // Handle the end of the run.
  const handleEnd = (event: StreamEvent) => {
    console.log('handleEnd', event);

    onComplete();
  };

  return { processEvents };
};
