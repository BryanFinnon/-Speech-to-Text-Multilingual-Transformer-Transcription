import { AudioManager } from "./components/AudioManager";
import Transcript from "./components/Transcript";
import { useTranscriber } from "./hooks/useTranscriber";

function App() {
    const transcriber = useTranscriber();

    return (
        <div className='flex justify-center items-center min-h-screen'>
            <div className='container flex flex-col justify-center items-center'>
                <h1 className='text-5xl font-extrabold tracking-tight text-slate-900 sm:text-7xl text-center'>
                Transcription Multilingues
                </h1>
                <h2 className='mt-3 mb-5 px-4 text-center text-1xl font-semibold tracking-tight text-slate-900 sm:text-2xl'>
                Transformez vos enregistrements en texte en un clin d'œil avec Whisper
                </h2>
                <AudioManager transcriber={transcriber} />
                <Transcript transcribedData={transcriber.output} />
            </div>

            <div className='absolute bottom-4'>
                Made by{" "}
                <a
                    className='underline'
                    href='https://github.com/Angejules123'>
                    Tia Ange jules & Finon Bryan 
                </a>
                <a
                    className='underline'
                    href='https://scholar.google.com/citations?user=E26mCFgAAAAJ'>
                    (Supervised by S.Jaballi 
                </a>

                <a
                    className='underline'
                    href='https://scholar.google.com/citations?user=8P1tpUsAAAAJ'>
                    & Pr M.Zrigui )
                </a>
            </div>
        </div>
    );
}

export default App;
