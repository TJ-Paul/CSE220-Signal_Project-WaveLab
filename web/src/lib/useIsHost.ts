import { useEffect, useState } from 'react'
import { api } from './api'

let pending: Promise<boolean> | null = null

/** True when this browser runs on the presenter's machine; guests on the
 *  network get false, so controls that remove session data can be hidden. */
export function useIsHost(): boolean {
  const [isHost, setIsHost] = useState(false)
  useEffect(() => {
    pending ??= api
      .client()
      .then((c) => c.isHost)
      .catch(() => {
        pending = null
        return false
      })
    let live = true
    void pending.then((value) => live && setIsHost(value))
    return () => {
      live = false
    }
  }, [])
  return isHost
}
