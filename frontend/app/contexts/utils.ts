export function nodeToStep(node: string) {
  switch (node) {
    case "analyze_and_route_query":
      return 0;
    case "create_research_plan":
      return 1;
    case "conduct_research":
      return 2;
    case "respond":
      return 3;
    default:
      return 0;
  }
}

export function addDocumentLinks(
  text: string,
  inputDocuments: Record<string, any>[],
): string {
  return text.replace(/\[(\d+)\]/g, (match, number) => {
    const index = parseInt(number, 10);
    if (index >= 0 && index < inputDocuments.length) {
      const document = inputDocuments[index];
      if (document && document.metadata && document.metadata.source) {
        return `[[${number}]](${document.metadata.source})`;
      }
    }
    // Return the original match if no corresponding document is found
    return match;
  });
}
