import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import styles from './index.module.less'
import type { ChatMessage } from '@/types/chat'
import { Button } from '@/components/ui/button'
import { useEffect, useRef } from 'react'

interface ChatHistoryProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chatHistory: ChatMessage[];
  onDeleteHistory: () => void;
}

const ChatHistory = ({ open, onOpenChange, chatHistory, onDeleteHistory }: ChatHistoryProps) => {
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [open, chatHistory]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[80vw] h-[80vh] flex flex-col p-0">
        <DialogHeader className="p-6 pb-4 flex justify-between items-center">
          <DialogTitle>{t('history.title')}</DialogTitle>
          <Button variant="destructive" onClick={onDeleteHistory} className="ml-auto">{t('history.clear')}</Button>
        </DialogHeader>
        <ScrollArea className="flex-1 p-6 pt-0" ref={scrollRef}>
          <div className={styles.historyContainer}>
            {chatHistory.map((msg, index) => (
              <div
                key={index}
                className={cn(
                  styles.messageWrapper,
                  msg.role === 'user' ? styles.userMessageWrapper : styles.assistantMessageWrapper
                )}
              >
                <div 
                  className={cn(
                    styles.message,
                    msg.role === 'user' ? styles.userMessage : styles.assistantMessage
                  )}
                >
                  <div className={styles.messageContent}>{msg.role === 'user' ? t('history.user') : t('history.ai')}: {msg.content}</div>
                  <div className={styles.timestamp}>
                    {new Date(msg.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}

export default ChatHistory 