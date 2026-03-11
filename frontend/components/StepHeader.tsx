interface StepHeaderProps {
  step: number;
  title: string;
  subtitle: string;
}

export function StepHeader({ step, title, subtitle }: StepHeaderProps) {
  return (
    <div>
      <p className="text-[11px] font-medium text-steel-400 uppercase tracking-widest mb-1">
        Step {step}
      </p>
      <h2 className="font-serif text-2xl text-oxford leading-tight">{title}</h2>
      <p className="text-sm text-steel-500 mt-1.5 leading-relaxed max-w-2xl">{subtitle}</p>
    </div>
  );
}
