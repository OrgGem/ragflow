import { RAGFlowFormItem } from '@/components/ragflow-form';
import { ButtonLoading } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Form } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { RAGFlowSelect } from '@/components/ui/select';
import { LLMFactory } from '@/constants/llm';
import { IModalProps } from '@/interfaces/common';
import { VerifyResult } from '@/pages/user-setting/setting-model/hooks';
import { buildOptions } from '@/utils/form';
import { zodResolver } from '@hookform/resolvers/zod';
import { t } from 'i18next';
import { memo } from 'react';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';
import { LLMHeader } from '../../components/llm-header';
import VerifyButton from '../verify-button';

const FormSchema = z.object({
  llm_name: z.string().min(1, {
    message: t('setting.vietocr.modelNameRequired'),
  }),
  vietocr_device: z.string().default('cpu'),
  vietocr_model_name: z.string().default('vgg_transformer'),
});

export type VietOCRFormValues = z.infer<typeof FormSchema>;

const modelNameOptions = buildOptions([
  'vgg_transformer',
  'vgg_seq2seq',
  'resnet_transformer',
  'resnet_seq2seq',
]);

const deviceOptions = buildOptions(['cpu', 'cuda', 'cuda:0', 'cuda:1']);

const VietOCRModal = ({
  visible,
  hideModal,
  onOk,
  onVerify,
  loading,
}: IModalProps<VietOCRFormValues> & {
  onVerify?: (
    postBody: any,
  ) => Promise<boolean | void | VerifyResult | undefined>;
}) => {
  const { t } = useTranslation();

  const form = useForm<VietOCRFormValues>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      vietocr_device: 'cpu',
      vietocr_model_name: 'vgg_transformer',
    },
  });

  const handleOk = async (values: VietOCRFormValues) => {
    const ret = await onOk?.(values as any);
    if (ret) {
      hideModal?.();
    }
  };

  return (
    <Dialog open={visible} onOpenChange={hideModal}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            <LLMHeader name={LLMFactory.VietOCR} />
          </DialogTitle>
        </DialogHeader>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleOk)}
            className="space-y-6"
            id="vietocr-form"
          >
            <RAGFlowFormItem
              name="llm_name"
              label={t('setting.modelName')}
              required
            >
              <Input
                placeholder={t('setting.vietocr.modelNamePlaceholder')}
              />
            </RAGFlowFormItem>
            <RAGFlowFormItem
              name="vietocr_device"
              label={t('setting.vietocr.device')}
            >
              {(field) => (
                <RAGFlowSelect
                  value={field.value}
                  onChange={field.onChange}
                  options={deviceOptions}
                  placeholder={t('setting.vietocr.selectDevice')}
                />
              )}
            </RAGFlowFormItem>
            <RAGFlowFormItem
              name="vietocr_model_name"
              label={t('setting.vietocr.modelArch')}
            >
              {(field) => (
                <RAGFlowSelect
                  value={field.value}
                  onChange={field.onChange}
                  options={modelNameOptions}
                  placeholder={t('setting.vietocr.selectModelArch')}
                />
              )}
            </RAGFlowFormItem>
            {onVerify && (
              <VerifyButton
                onVerify={onVerify as (postBody: any) => Promise<VerifyResult>}
              />
            )}
          </form>
        </Form>
        <DialogFooter>
          <div className="flex gap-2">
            <ButtonLoading type="submit" form="vietocr-form" loading={loading}>
              {t('common.save', 'Save')}
            </ButtonLoading>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default memo(VietOCRModal);
