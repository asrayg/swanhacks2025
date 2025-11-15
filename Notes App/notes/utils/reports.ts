import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ClinicalReport {
  id: string;
  date: string;
  heartRate: number;
  bloodAlcohol: number;
  bloodPressure: string;
  height: string;
  content: string;
  createdAt: number;
}

const STORAGE_KEY = '@clinical_reports';

export async function saveReport(report: Omit<ClinicalReport, 'id' | 'createdAt'>): Promise<ClinicalReport> {
  try {
    const reports = await getReports();
    const newReport: ClinicalReport = {
      ...report,
      id: `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now(),
    };
    reports.push(newReport);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
    return newReport;
  } catch (error) {
    console.error('Error saving report:', error);
    throw error;
  }
}

export async function getReports(): Promise<ClinicalReport[]> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEY);
    if (data) {
      return JSON.parse(data);
    }
    return [];
  } catch (error) {
    console.error('Error getting reports:', error);
    return [];
  }
}

export async function getReport(id: string): Promise<ClinicalReport | null> {
  try {
    const reports = await getReports();
    return reports.find(r => r.id === id) || null;
  } catch (error) {
    console.error('Error getting report:', error);
    return null;
  }
}

export async function deleteReport(id: string): Promise<boolean> {
  try {
    const reports = await getReports();
    const filtered = reports.filter(r => r.id !== id);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    return true;
  } catch (error) {
    console.error('Error deleting report:', error);
    return false;
  }
}

