import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Globe } from 'lucide-react';

interface LanguageSelectorProps {
  className?: string;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({ className }) => {
  const { i18n, t } = useTranslation();
  const handleLanguageChange = (language: string) => {
    i18n.changeLanguage(language);
  };

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Globe className="h-4 w-4" />
      <Select value={i18n.language} onValueChange={handleLanguageChange}>
        <SelectTrigger className="w-32">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="zh">{t('common.chinese')}</SelectItem>
          <SelectItem value="en">{t('common.english')}</SelectItem>
          <SelectItem value="ja">{t('common.japanese')}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

export default LanguageSelector;