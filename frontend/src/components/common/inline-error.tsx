type InlineErrorProps = {
  message: string;
  onRetry?: () => void;
  retryLabel?: string;
  messageClassName?: string;
  retryClassName?: string;
};

export function InlineError({
  message,
  onRetry,
  retryLabel = "Retry",
  messageClassName = "inline-error__message",
  retryClassName = "inline-error__retry",
}: InlineErrorProps) {
  return (
    <div>
      <p className={messageClassName} role="alert">
        {message}
      </p>
      {onRetry ? (
        <button className={retryClassName} type="button" onClick={onRetry}>
          {retryLabel}
        </button>
      ) : null}
    </div>
  );
}
