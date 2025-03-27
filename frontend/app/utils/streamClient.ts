interface ClientProps {
  url: string;
  idToken?: string;
}

// A client for streaming events from a server.
export const streamClient = ({ url, idToken }: ClientProps) => {
  const logFeedback = async (
    feedbackDict: Record<string, any>,
    runId: string,
  ) => {
    const scoreMap: Record<string, number> = {
      '😞': 0.0,
      '🙁': 0.25,
      '😐': 0.5,
      '🙂': 0.75,
      '😀': 1.0,
    };

    const score = scoreMap[feedbackDict.score] ?? feedbackDict.score;
    const feedback = {
      ...feedbackDict,
      score,
      run_id: runId,
      log_type: 'feedback',
    };

    const feedbackUrl = new URL('feedback', url).toString();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (idToken) {
      headers.Authorization = `Bearer ${idToken}`;
    }

    await fetch(feedbackUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(feedback),
    });
  };

  const streamEvents = async function* (data: Record<string, any>) {
    const eventsUrl = new URL('stream_events', url).toString();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    };

    if (idToken) {
      headers.Authorization = `Bearer ${idToken}`;
    }

    // console.log('data', data);

    try {
      const response = await fetch(eventsUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const msg = await response.text();

        throw new Error(
          `Network response was not ok ${response.status}: ${msg}`,
        );
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is empty');
      }

      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          // Process any remaining buffer
          if (buffer.trim()) {
            try {
              yield JSON.parse(buffer);
            } catch (error) {
              console.error('Final buffer parse error:', buffer, error);
            }
          }
          break;
        }

        // Decode the chunk and combine with any previous buffer
        buffer += decoder.decode(value, { stream: true });

        // Split the buffer into lines
        const lines = buffer.split('\n');

        // Process complete lines
        while (lines.length > 1) {
          const line = lines.shift();
          if (line?.trim()) {
            try {
              yield JSON.parse(line);
            } catch (error) {
              console.error('Failed to parse line:', line, error);
            }
          }
        }

        // Keep the last (potentially incomplete) line in the buffer
        buffer = lines[0];
      }
    } catch (error) {
      console.error('Error streaming events:', error);
      // throw error; // Re-throw to allow caller to handle
    }
  };

  return { logFeedback, streamEvents };
};
