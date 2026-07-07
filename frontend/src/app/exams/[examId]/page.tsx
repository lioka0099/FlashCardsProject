import { ExamWorkspace } from "@/components/exam/study/exam-workspace";

type ExamPageProps = {
  params: Promise<{
    examId: string;
  }>;
};

export default async function ExamPage({ params }: ExamPageProps) {
  const { examId } = await params;
  return <ExamWorkspace examId={examId} />;
}
