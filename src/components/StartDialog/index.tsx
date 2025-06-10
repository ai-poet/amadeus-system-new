import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useTranslation } from 'react-i18next';
import styles from './index.module.less';

interface StartDialogProps {
  onStart: () => void;
  isFirstConfig?: boolean;
  isConnecting?: boolean;
}

const StartDialog: React.FC<StartDialogProps> = ({ onStart, isFirstConfig, isConnecting = false }) => {
  const { t } = useTranslation();
  
  return (
    <div className={styles.overlay}>
      <Card className={styles.card}>
        <h2 className="text-lg font-semibold mb-4">{t('dialog.ready')}</h2>
        <p className="text-sm text-muted-foreground mb-6">
          {isConnecting
            ? t('dialog.connecting')
            : isFirstConfig 
              ? t('dialog.firstConfigComplete')
              : t('dialog.readyToStart')}
        </p>
        <Button 
          onClick={onStart} 
          className="w-full"
          disabled={isConnecting}
        >
          {isConnecting ? t('dialog.connecting2') : t('dialog.startConversation')}
        </Button>
      </Card>
    </div>
  );
};

export default StartDialog; 