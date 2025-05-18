/**
 * EventStreamParser - A utility for parsing Server-Sent Events (SSE) streams
 * and converting the data to JSON objects.
 */

/**
 * Event interface representing a parsed SSE event
 */
export interface ParsedEvent {
    id?: string;
    event?: string;
    data: any;
    retry?: number;
  }
  
  /**
   * Options for the EventStreamParser
   */
  export interface EventStreamParserOptions {
    /** 
     * Whether to automatically parse JSON data (default: true)
     * If false, data will be returned as raw strings
     */
    parseJson?: boolean;
    /** 
     * Custom error handler for JSON parsing errors
     * By default, parsing errors are logged to console and the raw string is returned
     */
    onJsonError?: (error: Error, data: string) => any;
  }
  
  /**
   * EventStreamParser class for parsing SSE streams and yielding JSON data
   */
  export class EventStreamParser {
    private buffer: string = '';
    private options: EventStreamParserOptions;
  
    /**
     * Creates a new EventStreamParser
     * @param options Configuration options
     */
    constructor(options: EventStreamParserOptions = {}) {
      this.options = {
        parseJson: true,
        onJsonError: (error, data) => {
          console.error('Failed to parse JSON data:', error);
          return data; // Return the raw string if parsing fails
        },
        ...options
      };
    }
  
    /**
     * Parse a chunk of data from an event stream
     * @param chunk New chunk of data to parse
     * @returns Array of parsed events
     */
    parse(chunk: string): ParsedEvent[] {
      // Append the new chunk to our buffer
      this.buffer += chunk;
  
      // Split on double newlines which indicate the end of an event
      const events: ParsedEvent[] = [];
      const parts = this.buffer.split(/\r\n\r\n|\n\n/);
      
      // The last part might be incomplete, so we keep it in the buffer
      this.buffer = parts.pop() || '';
  
      // Process each complete event
      for (const part of parts) {
        const event = this.parseEvent(part);
        if (event) {
          events.push(event);
        }
      }
  
      return events;
    }
  
    /**
     * Parse a single event from the event stream
     * @param eventStr Raw event string
     * @returns Parsed event object or null if invalid
     */
    private parseEvent(eventStr: string): ParsedEvent | null {
      if (!eventStr.trim()) {
        return null; // Skip empty events
      }
  
      // Parse the event string line by line
      const event: ParsedEvent = { data: '' };
      let dataLines: string[] = [];
  
      const lines = eventStr.split(/\r\n|\n/);
      for (const line of lines) {
        // Skip comments and empty lines
        if (!line || line.startsWith(':')) {
          continue;
        }
  
        // Extract field and value
        const colonIndex = line.indexOf(':');
        const field = colonIndex > 0 ? line.slice(0, colonIndex) : line;
        const value = colonIndex > 0 ? line.slice(colonIndex + 1).trimStart() : '';
  
        // Process each field type
        switch (field) {
          case 'event':
            event.event = value;
            break;
          case 'data':
            dataLines.push(value);
            break;
          case 'id':
            if (!value.includes('\u0000')) { // Null character check
              event.id = value;
            }
            break;
          case 'retry':
            const retry = parseInt(value, 10);
            if (!isNaN(retry)) {
              event.retry = retry;
            }
            break;
        }
      }
  
      // Join data lines with newlines
      const rawData = dataLines.join('\n');
      
      // Parse JSON data if enabled
      if (this.options.parseJson && rawData) {
        try {
          event.data = JSON.parse(rawData);
        } catch (error) {
          if (this.options.onJsonError && error instanceof Error) {
            event.data = this.options.onJsonError(error, rawData);
          } else {
            event.data = rawData;
          }
        }
      } else {
        event.data = rawData;
      }
  
      return event;
    }
  
    /**
     * Clear the internal buffer
     */
    reset(): void {
      this.buffer = '';
    }
  }
  
  /**
   * Create an async generator that consumes an event stream and yields parsed events
   * @param response Fetch Response object with content-type text/event-stream
   * @param options Parser options
   * @returns Async generator yielding parsed events
   */
  export async function* streamEvents(
    response: Response,
    options?: EventStreamParserOptions
  ): AsyncGenerator<ParsedEvent, void, unknown> {
    if (!response.body) {
      throw new Error('Response has no body');
    }
  
    // Verify this is an event stream
    const contentType = response.headers.get('content-type');
    if (!contentType?.includes('text/event-stream')) {
      throw new Error(`Expected content-type text/event-stream but received ${contentType}`);
    }
  
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const parser = new EventStreamParser(options);
  
    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
  
        const chunk = decoder.decode(value, { stream: true });
        const events = parser.parse(chunk);
        
        for (const event of events) {
          yield event;
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
  
