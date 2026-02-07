"use client";

import { useEffect, useRef } from "react";

export function DebugStyles() {
  const bodyRef = useRef<HTMLBodyElement | null>(null);
  
  useEffect(() => {
    // #region agent log
    const checkStyles = () => {
      const body = document.body;
      const html = document.documentElement;
      const computed = window.getComputedStyle(body);
      const htmlComputed = window.getComputedStyle(html);
      
      // Check if Tailwind classes exist
      const testEl = document.createElement('div');
      testEl.className = 'bg-steel-200 bg-oxford';
      document.body.appendChild(testEl);
      const testComputed = window.getComputedStyle(testEl);
      document.body.removeChild(testEl);
      
      fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'debug-styles.tsx:20',message:'Style diagnostics',data:{bodyBg:computed.backgroundColor,bodyFont:computed.fontFamily,htmlFont:htmlComputed.fontFamily,bodyClasses:body.className,htmlClasses:html.className,testElBg:testComputed.backgroundColor,hasSteel200:testComputed.backgroundColor.includes('226')||testComputed.backgroundColor.includes('232')},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      
      // Check if CSS file loaded
      const stylesheets = Array.from(document.styleSheets);
      const globalsCss = stylesheets.find(sheet => {
        try {
          return sheet.href?.includes('globals') || sheet.href?.includes('_app');
        } catch {
          return false;
        }
      });
      
      fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'debug-styles.tsx:35',message:'Stylesheet check',data:{totalSheets:stylesheets.length,hasGlobals:!!globalsCss,globalsHref:globalsCss?.href},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
    };
    
    // Run after a short delay to ensure styles are applied
    setTimeout(checkStyles, 100);
    // #endregion
  }, []);
  
  return null;
}
