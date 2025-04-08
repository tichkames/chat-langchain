"use client";

import { useEffect, useState } from "react";

import { useQueryState } from "nuqs";

export function useThreads(userId: string | undefined) {
  const [isUserThreadsLoading, setIsUserThreadsLoading] = useState(false);
  const [userThreads, setUserThreads] = useState<any[]>([]);
  const [threadId, setThreadId] = useQueryState("threadId");

  useEffect(() => {
    if (typeof window == "undefined" || !userId) return;
    getUserThreads(userId);
  }, [userId]);

  const getUserThreads = async (id: string) => {
    setIsUserThreadsLoading(true);
    try {
      const data = await fetch(`http://localhost:8000/threads/${id}`);
      const userThreads = await data.json();
      console.log('userThreads', userThreads);

      if (userThreads.length > 0) {
        const lastInArray = userThreads[0];
        const allButLast = userThreads.slice(1, userThreads.length);
        const filteredThreads = allButLast.filter(
          (thread: any) => thread.values && Object.keys(thread.values).length > 0,
        );
        setUserThreads([...filteredThreads, lastInArray]);
      }
    } finally {
      setIsUserThreadsLoading(false);
    }
  };

  const getThreadById = async (id: string) => {
    const data = await fetch(`http://localhost:8001/threads/${id}`);
    const state = await data.json();
    console.log('state', state);
  
    return state;
  };

  const deleteThread = async (id: string, clearMessages: () => void) => {
    if (!userId) {
      throw new Error("User ID not found");
    }
    setUserThreads((prevThreads) => {
      const newThreads = prevThreads.filter(
        (thread) => thread.thread_id !== id,
      );
      return newThreads;
    });

    // await client.threads.delete(id);

    if (id === threadId) {
      // Remove the threadID from query params, and refetch threads to
      // update the sidebar UI.
      clearMessages();
      getUserThreads(userId);
      setThreadId(null);
    }
  };

  return {
    isUserThreadsLoading,
    userThreads,
    getThreadById,
    setUserThreads,
    getUserThreads,
    deleteThread,
  };
}
