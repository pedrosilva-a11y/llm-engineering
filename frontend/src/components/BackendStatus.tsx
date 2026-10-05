import { useEffect, useState } from 'react'

import { getBackendHealth } from '../api/health'

import './BackendStatus.css'

const POLL_INTERVAL = 10_000
type BackendState = 'checking' | 'online' | 'offline'

export function BackendStatus() {
  const [status, setStatus] = useState<BackendState>('checking')
  const [service, setService] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    async function checkBackend(): Promise<void> {
      try {
        const health = await getBackendHealth()

        if (isMounted) {
          setStatus('online')
          setService(health.service)
        }
      } catch {
        if (isMounted) {
          setStatus('offline')
          setService(null)
        }
      }
    }

    void checkBackend()

    const intervalId = window.setInterval(() => {
      void checkBackend()
    }, POLL_INTERVAL)

    return () => {
      isMounted = false
      window.clearInterval(intervalId)
    }
  }, [])

  const label =
    status === 'checking'
      ? 'Backend checking'
      : status === 'online'
        ? 'Backend online'
        : 'Backend offline'

  return (
    <div
      className={`backend-status backend-status--${status}`}
      title={service ?? label}
      aria-label={label}
    >
      <span className="backend-status__dot" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}
