#curious meme on guidance systems
# https://www.youtube.com/watch?v=c94bALAgENM
import numpy as np

class TERCOM(object):
    #the missile knows where it is at all times
    #it knows this because it knows where it isn't->
    def __init__(contour_maps):
        self.contour_maps = contour_maps
    
    def radar_integrate(self):
        
    def guidance_system_event_loop():
        self.radar_integrate()
        variation = self.compute_variation()
        self.flight_track[t] = get_current_flight_track()
        
        #it obtains a difference or deviation
        deviation = max(#whichever is greater,
            get_linear_distance(flight_course[t+1], #->by subtracting where it is from where it isn't
            flight_track[t]), get_linear_distance(flight_track[t], flight_course[t+1])) #or from where it isn't from where it is
        
        #the guidance system uses deviations to generate corrrective commands to drive the missile
        #from a position where it is (flight_track[t]) to a position where it isn't (flight_course[t+1])
        #arriving at a position where it wasn't, but is now(flight_track[t] == flight_course[t])
        error_signal = variation + deviation
        corrective_command = CorrectiveCommand(error_signal, flight_track, flight_course)
        #consequently, the position where it is, is the position where it wasn't
        #so it follows the position where it was, is the position where it isn't
        
    def compute_variation(flight_track, flight_course):
        #in the event that the position where it is, is not the position where it wasn't
        #the system has aquired a variation
        if flight_track != flight_course:
            #the variation being the difference between where the missile is, and where the missile wasn't
            return compute_linear_distance(flight_track[t], flight_course[t])
        return 0
    
    def execute_corrective_command(cc):
        #if the variation is considered to be a significant factor
        if cc.variation >= self.VARIATION_THRESHOLD:
            self.variation_correct()
        #the missile guidance logic system will allow for the variation
        #provided the missile knows where it was or is not now (TERCOM())
        #due to the variation modifying some of the information obtained by the missile, it is not sure where it is
        
        #however the thought process of the missile is that it is sure where it isn't and it knows where it was
        #it now subtracts where it should be from where it wasn't and adds the variable that (compute variation) obtained by subtracting where it isn't from where it was
        #(guidance_system_event_loop)
        #in guidance system language, this is called error, or the difference between deviation and variation found in the algebraic difference, 
        # found between where the missile shouldn't be and where it is
        
    def compute_linear_distance(flight_track_pos, flight_course_pot)
        distance_vector = np.array([flight_track_pos - flight_course_pot, flight_track_pos - flight_course_pot])
        return np.linalg.norm(distance_vector)