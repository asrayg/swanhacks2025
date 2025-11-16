import { createClient } from '@supabase/supabase-js';

const SUPABASE_PROJECT_ID = 'lfvpbnzpsxxgppnlzhln';
const SUPABASE_URL = `https://${SUPABASE_PROJECT_ID}.supabase.co`;
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxmdnBibnpwc3h4Z3Bwbmx6aGxuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMyMzM4OTAsImV4cCI6MjA3ODgwOTg5MH0.PnYZTH0VGOmp_qZAH-s8xZFE64qeOypZsIks50M-pP4';
const SUPABASE_SERVICE_ROLE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxmdnBibnpwc3h4Z3Bwbmx6aGxuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzIzMzg5MCwiZXhwIjoyMDc4ODA5ODkwfQ.W8W03HO2-uRUrwBqfDFnd203s1NMUkgYiI3oRlPqHBg';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

export const supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

