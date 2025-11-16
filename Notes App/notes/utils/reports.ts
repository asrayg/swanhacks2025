import { supabase, supabaseAdmin } from './supabase';

export interface ClinicalReport {
  id: string;
  date: string;
  heartRate: number;
  bloodAlcohol: number;
  bloodPressure: string;
  height: string;
  content: string;
  video?: string;
  createdAt: number;
}

function extractDateFromContent(content: string): string {
  const dateMatch = content.match(/Report Generated: ([\d\s:-]+)/);
  if (dateMatch) {
    return dateMatch[1];
  }
  return new Date().toLocaleString();
}

function extractVitalsFromContent(content: string): {
  heartRate: number;
  bloodAlcohol: number;
  bloodPressure: string;
  height: string;
} {
  const heartRateMatch = content.match(/Heart Rate[:\s]+(\d+)/i);
  const bloodOxygenMatch = content.match(/Blood Oxygen[:\s]+(\d+)/i);
  const bloodPressureMatch = content.match(/Blood Pressure[:\s]+([\d/]+)/i);
  const heightMatch = content.match(/Height[:\s]+([^(\n]+)/i);

  let height = heightMatch ? heightMatch[1].trim() : '5\'10"';
  height = height.replace(/^\*\*\s+/, ''); 

  return {
    heartRate: heartRateMatch ? parseInt(heartRateMatch[1]) : 85,
    bloodAlcohol: bloodOxygenMatch ? parseInt(bloodOxygenMatch[1]) : 95,
    bloodPressure: bloodPressureMatch ? bloodPressureMatch[1] : '120/80',
    height: height,
  };
}

export async function saveReport(report: Omit<ClinicalReport, 'id' | 'createdAt'>): Promise<ClinicalReport> {
  const newReport: ClinicalReport = {
    ...report,
    id: `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    createdAt: Date.now(),
  };
  return newReport;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number = 10000): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => {
      reject(new Error('Request timeout'));
    }, timeoutMs);
  });
  
  return Promise.race([promise, timeoutPromise]);
}

export async function getReports(): Promise<ClinicalReport[]> {
  try {
    
    const queryPromise = supabase
      .from('Expo')
      .select('*')
      .order('created_at', { ascending: false });
    
    const { data, error } = await withTimeout(queryPromise, 10000);

    if (error) {
      console.error('Supabase error:', error);
      console.error('Error code:', error.code);
      console.error('Error message:', error.message);
      console.error('Error details:', JSON.stringify(error, null, 2));
      console.error('Error hint:', error.hint);
      
      if (error.code === 'PGRST301' || error.message?.includes('permission denied') || error.message?.includes('RLS')) {
        console.error('RLS (Row Level Security) may be blocking access.');
      }
      
      if (error.code === 'PGRST116' || error.message?.includes('relation') || error.message?.includes('does not exist')) {
        console.log('Trying lowercase table name "expo"...');
        const { data: dataLower, error: errorLower } = await supabase
          .from('expo')
          .select('*');
        
        if (errorLower) {
          console.error('Error with lowercase table name:', errorLower);
          return [];
        }
        
        if (!dataLower) {
          return [];
        }
        
        const reports: ClinicalReport[] = dataLower.map((row: any) => {
          const reportContent = row.Report || row.report || '';
          const videoUrl = row.Video || row.video || undefined;
          const vitals = extractVitalsFromContent(reportContent);
          const date = row.created_at 
            ? new Date(row.created_at).toLocaleString()
            : extractDateFromContent(reportContent);

          return {
            id: row.id?.toString() || `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            date: date,
            heartRate: vitals.heartRate,
            bloodAlcohol: vitals.bloodAlcohol,
            bloodPressure: vitals.bloodPressure,
            height: vitals.height,
            content: reportContent,
            video: videoUrl,
            createdAt: row.created_at 
              ? new Date(row.created_at).getTime() 
              : Date.now(),
          };
        });
        
        return reports;
      }
      
      return [];
    }

    console.log('Supabase response data:', JSON.stringify(data, null, 2));
    console.log('Number of records:', data?.length || 0);
    
    if (data && data.length === 0 && !error) {
      console.warn('Empty array returned - likely RLS (Row Level Security) blocking access');
      console.warn('Testing with service role key to confirm...');
      
      try {
        const { data: adminData, error: adminError } = await supabaseAdmin
          .from('Expo')
          .select('*')
          .order('created_at', { ascending: false });
        
        if (adminError) {
          console.error('Service role also failed:', adminError);
        } else if (adminData && adminData.length > 0) {
          const reports: ClinicalReport[] = adminData.map((row: any, index: number) => {
            console.log(`Processing admin row ${index}:`, {
              id: row.id,
              hasReport: !!(row.Report || row.report),
              hasVideo: !!(row.Video || row.video),
              created_at: row.created_at,
              allKeys: Object.keys(row)
            });
            
            const reportContent = row.Report || row.report || row.REPORT || row['Report'] || row['report'] || '';
            const videoUrl = row.Video || row.video || row.VIDEO || row['Video'] || row['video'] || undefined;
            const vitals = extractVitalsFromContent(reportContent);
            const date = row.created_at 
              ? new Date(row.created_at).toLocaleString()
              : extractDateFromContent(reportContent);

            const report: ClinicalReport = {
              id: row.id?.toString() || `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              date: date,
              heartRate: vitals.heartRate,
              bloodAlcohol: vitals.bloodAlcohol,
              bloodPressure: vitals.bloodPressure,
              height: vitals.height,
              content: reportContent,
              video: videoUrl,
              createdAt: row.created_at 
                ? new Date(row.created_at).getTime() 
                : Date.now(),
            };
            
            return report;
          });
          
          return reports;
        }
      } catch (adminErr) {
        console.error('Error testing with service role:', adminErr);
      }
      
      console.log('Data array is empty - likely RLS blocking access');
      return [];
    }
    
    if (data && data.length > 0) {
      console.log('First record keys:', Object.keys(data[0]));
      console.log('First record:', JSON.stringify(data[0], null, 2));
      console.log('First record Report field:', data[0].Report || data[0].report || 'NOT FOUND');
      console.log('First record Video field:', data[0].Video || data[0].video || 'NOT FOUND');
    }

    if (!data) {
      console.log('No data returned from Supabase');
      return [];
    }

    const reports: ClinicalReport[] = data.map((row: any, index: number) => {
      console.log(`Processing row ${index}:`, {
        id: row.id,
        hasReport: !!(row.Report || row.report),
        hasVideo: !!(row.Video || row.video),
        created_at: row.created_at,
        allKeys: Object.keys(row)
      });
      
      const reportContent = row.Report || row.report || row.REPORT || row['Report'] || row['report'] || '';
      const videoUrl = row.Video || row.video || row.VIDEO || row['Video'] || row['video'] || undefined;
      const vitals = extractVitalsFromContent(reportContent);
      const date = row.created_at 
        ? new Date(row.created_at).toLocaleString()
        : extractDateFromContent(reportContent);

      const report: ClinicalReport = {
        id: row.id?.toString() || `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        date: date,
        heartRate: vitals.heartRate,
        bloodAlcohol: vitals.bloodAlcohol,
        bloodPressure: vitals.bloodPressure,
        height: vitals.height,
        content: reportContent,
        video: videoUrl,
        createdAt: row.created_at 
          ? new Date(row.created_at).getTime() 
          : Date.now(),
      };
      
      console.log(`Mapped report ${index}:`, {
        id: report.id,
        date: report.date,
        contentLength: report.content.length,
        hasVideo: !!report.video
      });
      
      return report;
    });
    
    console.log('Total reports mapped:', reports.length);

    return reports;
  } catch (error) {
    console.error('Error getting reports:', error);
    return [];
  }
}

export async function getReport(id: string): Promise<ClinicalReport | null> {
  try {
    console.log('Fetching report with id:', id);
    
    let { data, error } = await supabase
      .from('Expo')
      .select('*')
      .eq('id', id)
      .single();

    if (error || !data) {
      console.warn('Anon key failed, trying service role key...');
      const adminResult = await supabaseAdmin
        .from('Expo')
        .select('*')
        .eq('id', id)
        .single();
      
      if (adminResult.error) {
        console.error('Error fetching report from Supabase (admin):', adminResult.error);
        return null;
      }
      
      data = adminResult.data;
      error = null;
    }

    if (!data) {
      console.log('No data found for report id:', id);
      return null;
    }

    console.log('Report data found:', {
      id: data.id,
      hasReport: !!(data.Report || data.report),
      hasVideo: !!(data.Video || data.video),
      allKeys: Object.keys(data)
    });

    const reportContent = data.Report || data.report || data.REPORT || data['Report'] || data['report'] || '';
    const videoUrl = data.Video || data.video || data.VIDEO || data['Video'] || data['video'] || undefined;
    const vitals = extractVitalsFromContent(reportContent);
    const date = data.created_at 
      ? new Date(data.created_at).toLocaleString()
      : extractDateFromContent(reportContent);

    return {
      id: data.id?.toString() || id,
      date: date,
      heartRate: vitals.heartRate,
      bloodAlcohol: vitals.bloodAlcohol,
      bloodPressure: vitals.bloodPressure,
      height: vitals.height,
      content: reportContent,
      video: videoUrl,
      createdAt: data.created_at 
        ? new Date(data.created_at).getTime() 
        : Date.now(),
    };
  } catch (error) {
    console.error('Error getting report:', error);
    return null;
  }
}

export async function deleteReport(id: string): Promise<boolean> {
  try {
    const { error } = await supabase
      .from('Expo')
      .delete()
      .eq('id', id);

    if (error) {
      console.error('Error deleting report from Supabase:', error);
      return false;
    }

    return true;
  } catch (error) {
    console.error('Error deleting report:', error);
    return false;
  }
}

