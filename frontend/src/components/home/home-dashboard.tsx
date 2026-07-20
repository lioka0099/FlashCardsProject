"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, LogOut, User } from "lucide-react";
import { ExamHistorySidebar } from "@/components/home/exam-history-sidebar";
import { UploadExamForm } from "@/components/home/upload-exam-form";
import { getMe, logout } from "@/lib/api/auth";
import "@/components/home/home-screen.css";

const DECK_NAME_INPUT_ID = "dash-deck-name";

export function HomeDashboard() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAccountOpen, setIsAccountOpen] = useState(false);
  const accountRef = useRef<HTMLDivElement | null>(null);
  const { data: me } = useQuery({ queryKey: ["me"], queryFn: getMe });
  const firstName = me?.name?.trim().split(/\s+/)[0];

  useEffect(() => {
    if (!isAccountOpen) {
      return;
    }
    function onPointerDown(event: MouseEvent) {
      if (accountRef.current?.contains(event.target as Node)) {
        return;
      }
      setIsAccountOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [isAccountOpen]);

  function focusDeckName() {
    setIsSidebarOpen(false);
    const input = document.getElementById(DECK_NAME_INPUT_ID);
    input?.scrollIntoView({ behavior: "smooth", block: "center" });
    (input as HTMLInputElement | null)?.focus();
  }

  return (
    <div className="dash">
      <button
        className="dash__toggle"
        type="button"
        onClick={() => setIsSidebarOpen((open) => !open)}
        aria-expanded={isSidebarOpen}
        aria-controls="home-sidebar"
      >
        {isSidebarOpen ? "Close tests" : "My Tests"}
      </button>

      <ExamHistorySidebar
        className={`dash-side${isSidebarOpen ? " dash-side--open" : ""}`}
        onNavigate={() => setIsSidebarOpen(false)}
        onNewTest={focusDeckName}
      />

      <div className="dash-main">
        <header className="dash-top">
          <div>
            <h1 className="dash-top__welcome">
              {firstName ? `Welcome back, ${firstName}! 👋` : "Welcome back! 👋"}
            </h1>
            <p className="dash-top__sub">Let&rsquo;s turn documents into understanding.</p>
          </div>

          <div className="dash-account" ref={accountRef}>
            <button
              className="dash-account__btn"
              type="button"
              aria-haspopup="menu"
              aria-expanded={isAccountOpen}
              onClick={() => setIsAccountOpen((open) => !open)}
              data-tour="account"
            >
              <span className="dash-account__avatar" aria-hidden="true">
                <User size={18} />
              </span>
              <span>Account</span>
              <ChevronDown size={16} />
            </button>

            <AnimatePresence>
              {isAccountOpen ? (
                <motion.div
                  className="dash-account__menu"
                  role="menu"
                  aria-label="Account options"
                  initial={{ opacity: 0, y: 8, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: "easeOut" }}
                >
                  <Link className="dash-account__link" href="/profile" role="menuitem">
                    Profile
                  </Link>
                  <Link className="dash-account__link" href="/settings" role="menuitem">
                    Settings
                  </Link>
                  <div className="dash-account__divider" role="separator" />
                  <button
                    className="dash-account__link dash-account__link--danger"
                    type="button"
                    role="menuitem"
                    onClick={logout}
                  >
                    <LogOut size={15} />
                    Log out
                  </button>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        </header>

        <main className="dash-content">
          <UploadExamForm deckNameInputId={DECK_NAME_INPUT_ID} />
        </main>
      </div>
    </div>
  );
}
