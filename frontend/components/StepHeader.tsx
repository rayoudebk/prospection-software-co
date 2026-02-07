"use client";

import { LucideIcon } from "lucide-react";

interface StepHeaderProps {
  icon: LucideIcon;
  step: number;
  title: string;
  subtitle: string;
}

export function StepHeader({ icon: Icon, step, title, subtitle }: StepHeaderProps) {
  return (
    <div className="bg-white border border-steel-200 p-5 mb-6">
      <div className="flex items-start gap-4">
        <div className="flex items-center justify-center w-12 h-12 bg-oxford text-white shrink-0">
          <Icon className="w-6 h-6" />
        </div>
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-steel-400 uppercase tracking-wider">
              Step {step}
            </span>
          </div>
          <h2 className="text-xl font-semibold text-oxford">{title}</h2>
          <p className="text-steel-500 mt-1">{subtitle}</p>
        </div>
      </div>
    </div>
  );
}
