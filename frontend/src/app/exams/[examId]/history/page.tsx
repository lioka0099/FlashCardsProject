import { HistoryList } from "@/components/exam/history/history-list";

type ExamHistoryPageProps = {
  params: Promise<{
    examId: string;
  }>;
};

export default async function ExamHistoryPage({ params }: ExamHistoryPageProps) {
  const { examId } = await params;
  return <HistoryList examId={examId} />;
}
