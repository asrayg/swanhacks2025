import { ScrollView, StyleSheet, View, TouchableOpacity, Pressable } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/theme';
import { getReport, ClinicalReport } from '@/utils/reports';
import { VideoView, useVideoPlayer } from 'expo-video';

export default function ReportDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const [report, setReport] = useState<ClinicalReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReport();
  }, [id]);

  async function loadReport() {
    if (!id) return;
    try {
      const loadedReport = await getReport(id);
      setReport(loadedReport);
    } catch (error) {
      console.error('Error loading report:', error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <ThemedView style={styles.container}>
        <ThemedText>Loading...</ThemedText>
      </ThemedView>
    );
  }

  if (!report) {
    return (
      <ThemedView style={styles.container}>
        <ThemedText>Report not found</ThemedText>
      </ThemedView>
    );
  }

  return (
    <ScrollView 
      style={[styles.container, { backgroundColor: isDark ? Colors.dark.background : Colors.light.background }]}
      contentContainerStyle={styles.contentContainer}
    >
      <View style={styles.header}>
        <TouchableOpacity 
          onPress={() => router.back()}
          style={styles.backButton}
        >
          <IconSymbol name="chevron.left" size={24} color={isDark ? Colors.dark.tint : Colors.light.tint} />
          <ThemedText style={styles.backButtonText}>Back</ThemedText>
        </TouchableOpacity>
      </View>

      <ThemedView style={styles.headerContent}>
        <ThemedText type="title" style={styles.title}>Clinical Report</ThemedText>
        <ThemedText style={styles.subtitle}>{report.date}</ThemedText>
      </ThemedView>

      <View style={styles.vitalsContainer}>
        <VitalCard
          icon="heart.fill"
          label="Heart Rate"
          value={report.heartRate.toString()}
          unit="bpm"
          color="#E74C3C"
          isDark={isDark}
        />
        <VitalCard
          icon="testtube.2"
          label="Blood Oxygen"
          value={report.bloodAlcohol.toString()}
          unit="%"
          color="#F39C12"
          isDark={isDark}
        />
        <VitalCard
          icon="waveform.path.ecg"
          label="Blood Pressure"
          value={report.bloodPressure || "120/80"}
          unit="mm"
          color="#9B59B6"
          isDark={isDark}
        />
        <VitalCard
          icon="ruler.fill"
          label="Height"
          value={report.height}
          unit=""
          color="#3498DB"
          isDark={isDark}
        />
      </View>

      <ThemedView style={styles.reportContainer}>
        <ThemedText type="subtitle" style={styles.reportTitle}>Clinical Examination Report</ThemedText>
        <ThemedView style={styles.reportContent}>
          <ThemedText style={styles.reportText}>{report.content}</ThemedText>
        </ThemedView>
      </ThemedView>

      {report.video && <VideoPlayer videoUrl={report.video} isDark={isDark} />}
    </ScrollView>
  );
}

function VideoPlayer({ videoUrl, isDark }: { videoUrl: string; isDark: boolean }) {
  const player = useVideoPlayer(videoUrl, (player) => {
    player.loop = false;
    // Don't auto-play - let user control playback
  });

  return (
    <ThemedView style={styles.videoContainer}>
      <ThemedText type="subtitle" style={styles.videoTitle}>Examination Video</ThemedText>
      <View style={styles.videoWrapper}>
        <VideoView
          player={player}
          style={styles.video}
          allowsFullscreen
          allowsPictureInPicture
          contentFit="contain"
          nativeControls
        />
      </View>
    </ThemedView>
  );
}

function VitalCard({ 
  icon, 
  label, 
  value, 
  unit, 
  color, 
  isDark 
}: { 
  icon: string; 
  label: string; 
  value: string; 
  unit: string; 
  color: string; 
  isDark: boolean;
}) {
  return (
    <View style={[
      styles.vitalCard,
      { 
        backgroundColor: isDark ? '#1E1E1E' : '#FFFFFF',
        shadowColor: isDark ? '#000' : '#000',
      }
    ]}>
      <View style={[styles.iconContainer, { backgroundColor: `${color}20` }]}>
        <IconSymbol name={icon} size={32} color={color} />
      </View>
      <ThemedText style={[
        styles.vitalLabel,
        (label === 'Blood Oxygen' || label === 'Blood Pressure') && styles.vitalLabelSmall
      ]}>{label}</ThemedText>
      <View style={styles.vitalValueContainer}>
        <ThemedText style={[
          styles.vitalValue, 
          { color },
          label === 'Blood Pressure' && styles.vitalValueSmall
        ]}>{value}</ThemedText>
        {unit && <ThemedText style={styles.vitalUnit}>{unit}</ThemedText>}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginTop: 55,
    marginBottom: 16,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  backButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  headerContent: {
    marginBottom: 24,
    alignItems: 'center',
  },
  title: {
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.7,
    textAlign: 'center',
  },
  vitalsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 32,
    gap: 12,
    flexWrap: 'wrap',
  },
  vitalCard: {
    flex: 1,
    minWidth: 100,
    padding: 20,
    borderRadius: 20,
    alignItems: 'center',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 5,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  iconContainer: {
    width: 64,
    height: 64,
    borderRadius: 32,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  vitalLabel: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 8,
    textAlign: 'center',
    fontWeight: '500',
  },
  vitalLabelSmall: {
    fontSize: 8.5,
  },
  vitalValueContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 4,
  },
  vitalValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  vitalValueSmall: {
    fontSize: 18,
  },
  vitalUnit: {
    fontSize: 14,
    opacity: 0.7,
    fontWeight: '600',
  },
  videoContainer: {
    marginBottom: 24,
  },
  videoTitle: {
    marginTop: 25,
    marginBottom: 16,
    textAlign: 'center',
  },
  videoWrapper: {
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 5,
  },
  video: {
    width: '100%',
    aspectRatio: 16 / 9,
  },
  reportContainer: {
    marginTop: 8,
  },
  reportTitle: {
    marginBottom: 16,
    textAlign: 'center',
  },
  reportContent: {
    padding: 24,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.1)',
    backgroundColor: 'rgba(0,0,0,0.02)',
  },
  reportText: {
    fontSize: 14,
    lineHeight: 22,
    fontFamily: 'monospace',
    letterSpacing: 0.3,
  },
});

