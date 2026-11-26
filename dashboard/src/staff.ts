// The signed-in staff member, as every screen sees them.
import { createContext, use } from "react";

import type { Api } from "./api";
import type { CustomerFeed } from "./feed";
import type { Role } from "./roles";

export interface Staff {
  email: string;
  role: Role;
  /** A guest (T411): no account, reading as a viewer. */
  guest: boolean;
  signOut: () => void;
  api: Api;
  /** Every customer, live (T403). */
  feed: CustomerFeed;
}

export const StaffContext = createContext<Staff | null>(null);

export function useStaff(): Staff {
  const staff = use(StaffContext);
  if (staff === null) throw new Error("Screens are shown only to signed-in staff");
  return staff;
}
