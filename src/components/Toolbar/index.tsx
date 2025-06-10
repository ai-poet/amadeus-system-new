import { Button } from '@/components/ui/button'
import { Mic, MicOff, Video, VideoOff, History, LogOut, Settings } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import styles from './index.module.less'

interface ToolbarProps {
  isListening: boolean;
  isVideoOn: boolean;
  onToggleListening: () => void;
  onToggleVideo: () => void;
  onShowHistory: () => void;
  onLogout: () => void;
  onShowConfig: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ isListening, isVideoOn, onToggleListening, onToggleVideo, onShowHistory, onLogout, onShowConfig }) => {
  const { t } = useTranslation();
  
  return (
    <div className={styles.toolbar}>
      <Button
        onClick={onToggleListening}
        variant="ghost"
        title={isListening ? t('toolbar.micOff') : t('toolbar.micOn')}
      >
        {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
      </Button>
      <Button
        onClick={onToggleVideo}
        variant="ghost"
        title={isVideoOn ? t('toolbar.videoOff') : t('toolbar.videoOn')}
      >
        {isVideoOn ? <VideoOff className="h-6 w-6" /> : <Video className="h-6 w-6" />}
      </Button>
      <Button
        onClick={onShowHistory}
        variant="ghost"
        title={t('toolbar.history')}
      >
        <History className="h-6 w-6" />
      </Button>
      <Button
        onClick={onShowConfig}
        variant="ghost"
        title={t('toolbar.config')}
      >
        <Settings className="h-6 w-6" />
      </Button>
      <Button
        onClick={onLogout}
        variant="ghost"
        title={t('toolbar.logout')}
      >
        <LogOut className="h-6 w-6" />
      </Button>
    </div>
  )
}

export default Toolbar 