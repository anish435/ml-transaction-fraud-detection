import { useState, useEffect, useCallback, useRef } from "react";

export function usePolling<T>(
  fetchFn: () => Promise<T>,
  intervalMs = 3000,
  enabled = true
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [secondsAgo, setSecondsAgo] = useState<number>(0);

  const fetchRef = useRef(fetchFn);
  fetchRef.current = fetchFn;

  const execute = useCallback(async () => {
    try {
      const result = await fetchRef.current();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
      setSecondsAgo(0);
    } catch (err: any) {
      setError(err.message || "Failed to fetch telemetry");
    } finally {
      setLoading(false);
    }
  }, []);

  // Polling loop
  useEffect(() => {
    if (!enabled) return;

    execute(); // initial fetch

    const timer = setInterval(() => {
      execute();
    }, intervalMs);

    return () => clearInterval(timer);
  }, [intervalMs, enabled, execute]);

  // Seconds ago timer
  useEffect(() => {
    const ticker = setInterval(() => {
      if (lastUpdated) {
        setSecondsAgo(Math.floor((Date.now() - lastUpdated.getTime()) / 1000));
      }
    }, 1000);

    return () => clearInterval(ticker);
  }, [lastUpdated]);

  return {
    data,
    loading,
    error,
    lastUpdated,
    secondsAgo,
    refresh: execute,
  };
}
