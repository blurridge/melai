import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Sun, Moon, Laptop } from "lucide-react";

export function ModeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <div className="flex items-center space-x-2">
      <button
        onClick={() => setTheme("light")}
        className={`p-2 rounded-md ${
          theme === "light" 
            ? "bg-primary/10 text-primary" 
            : "text-muted-foreground hover:text-foreground"
        }`}
        aria-label="Light mode"
        title="Light mode"
      >
        <Sun className="h-5 w-5" />
      </button>
      <button
        onClick={() => setTheme("dark")}
        className={`p-2 rounded-md ${
          theme === "dark" 
            ? "bg-primary/10 text-primary" 
            : "text-muted-foreground hover:text-foreground"
        }`}
        aria-label="Dark mode"
        title="Dark mode"
      >
        <Moon className="h-5 w-5" />
      </button>
      <button
        onClick={() => setTheme("system")}
        className={`p-2 rounded-md ${
          theme === "system" 
            ? "bg-primary/10 text-primary" 
            : "text-muted-foreground hover:text-foreground"
        }`}
        aria-label="System preference"
        title="System preference"
      >
        <Laptop className="h-5 w-5" />
      </button>
    </div>
  );
} 