import { useState } from 'react'
import { Check, Share2 } from 'lucide-react'
import { api } from '../lib/api'
import { Button } from './ui'

/** Publish an encoded vault file to the Shared Files tab, byte for byte. */
export function ShareWithRoom({ artifactId, className }: { artifactId: string; className?: string }) {
  const [state, setState] = useState<'idle' | 'busy' | 'done'>('idle')
  const [error, setError] = useState<string | null>(null)

  const share = async () => {
    setState('busy')
    setError(null)
    try {
      await api.shareVaultArtifact(artifactId)
      setState('done')
    } catch (e) {
      setError((e as Error).message)
      setState('idle')
    }
  }

  return (
    <div className={className}>
      <Button
        size="sm"
        icon={state === 'done' ? <Check size={13} /> : <Share2 size={13} />}
        loading={state === 'busy'}
        disabled={state === 'done'}
        onClick={() => void share()}
      >
        {state === 'done' ? 'Shared — in Shared Files' : 'Share with room'}
      </Button>
      {error && <p className="mt-1 text-[11px] text-danger">{error}</p>}
    </div>
  )
}
