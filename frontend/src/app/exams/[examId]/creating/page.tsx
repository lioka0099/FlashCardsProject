import { CreatingTestScreen } from "@/components/exam/creating/creating-test-screen";

type Props = { params: Promise<{ examId: string }> };

export default async function CreatingPage({ params }: Props) {
  const { examId } = await params;
  return <CreatingTestScreen examId={examId} />;
}
